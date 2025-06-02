#!/usr/bin/env python3

import rospy
import actionlib
import math
import numpy as np
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped, Quaternion, Point, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from robots_moviles.topology_library import GraphIO, PathPlanner
from std_srvs.srv import Empty

class TopologicalNavigation:
    def __init__(self):
        rospy.init_node('topological_navigation')
        
        # Parámetros
        self.map_resolution = rospy.get_param('~resolution', 0.05)  # m/pixel
        self.map_origin_x = rospy.get_param('~origin_x', -10.0)
        self.map_origin_y = rospy.get_param('~origin_y', -10.0)
        self.graph_file = rospy.get_param('~graph_file', 'grafo_optimizado_i.json.gz')
        
        # Estado del robot
        self.current_pose = None
        self.current_node = None
        self.graph_loaded = False
        self.move_base_ready = False

        # Parámetros de navegación
        self.move_base_timeout = rospy.Duration(120)
        self.max_node_distance = rospy.get_param('~max_node_distance', 0.2)
        
        # Conexiones ROS
        rospy.loginfo("Conectando con move_base...")
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        
        # Intentar conexión con move_base con timeout
        if not self.move_base_client.wait_for_server(rospy.Duration(10)):
            rospy.logerr("No se pudo conectar con move_base después de 10 segundos")
            rospy.signal_shutdown("move_base no disponible")
            return
        else:
            self.move_base_ready = True
            rospy.loginfo("Conexión con move_base establecida")

        # Topic para localización
        rospy.loginfo("Suscribiéndose a /amcl_pose...")
        self.pose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)

        # Cargar grafo
        rospy.loginfo("Cargando grafo topológico...")
        self.load_graph()

        # Esperar a tener al menos una localización antes de iniciar interfaz
        rospy.loginfo("Esperando primera localización...")
        timeout = rospy.Time.now() + rospy.Duration(20)
        while not rospy.is_shutdown() and self.current_pose is None and rospy.Time.now() < timeout:
            rospy.sleep(0.5)

        if self.current_pose is None:
            rospy.logerr("No se recibió localización después de 20 segundos")
            rospy.signal_shutdown("Sin datos de localización")
            return

        # Iniciar interfaz de usuario
        rospy.loginfo("Iniciando interfaz de usuario...")
        self.user_interface()

    def load_graph(self):
        """Carga el grafo optimizado desde archivo"""
        try:
            self.graph_io = GraphIO()
            self.optimized_graph = self.graph_io.load_graph(self.graph_file)
            self.planner = PathPlanner(self.optimized_graph)
            self.escenario_origen = self.optimized_graph['meta']['escenario_origen']
            self.graph_loaded = True
            
            # Verificar conversión de coordenadas
            for node_id, node_data in self.optimized_graph['nodes'].items():
                rospy.loginfo(f"Nodo {node_id}: ({node_data['x']}, {node_data['y']}) px -> ({node_data['x_m']:.2f}, {node_data['y_m']:.2f}) m")
        except Exception as e:
            rospy.logerr(f"Error al cargar el grafo: {str(e)}")
            self.graph_loaded = False

    def pose_callback(self, msg):
        """Actualiza la pose actual del robot"""
        if self.current_pose is None:
            rospy.loginfo("Primera localización recibida")
            
        self.current_pose = msg.pose.pose

        # Ajustar coordenadas (ejemplo con tus valores)
        self.current_pose.position.x += self.map_origin_x #4.5
        self.current_pose.position.y += self.map_origin_y #-6.0

        # Solo buscar nodo más cercano si el grafo está cargado
        if self.graph_loaded:
            current_x, current_y = (self.current_pose.position.x, self.current_pose.position.y)
            min_dist = float('inf')
            closest_node = None

            for node_id, node_data in self.optimized_graph['nodes'].items():
                dist = math.hypot(
                    node_data['x_m'] - current_x,
                    node_data['y_m'] - current_y
                )
                if dist < min_dist:
                    min_dist = dist
                    closest_node = node_id
            
            if closest_node != self.current_node:
                self.current_node = closest_node
                rospy.loginfo(f"Posición actual: nodo {closest_node} en ({current_x:.2f}, {current_y:.2f})")

    def calculate_orientation(self, start_orientation, target_orientation=None):
        """
        Calcula la orientación del robot.
        
        Args:
            start_orientation: Orientación inicial como vector [x,y]
            target_orientation: Orientación final deseada (opcional)
            
        Returns:
            Quaternion con la orientación resultante
        """
        if target_orientation:
            # Usar la orientación objetivo si se especifica
            yaw = math.atan2(target_orientation[1], target_orientation[0])
        else:
            # Calcular orientación basada en la dirección del movimiento
            if hasattr(self, 'goal_position'):
                dx = self.goal_position.x - self.current_pose.position.x
                dy = self.goal_position.y - self.current_pose.position.y
                yaw = math.atan2(dy, dx)
            else:
                # Usar orientación inicial si no hay objetivo
                yaw = math.atan2(start_orientation[1], start_orientation[0])
        
        return Quaternion(*quaternion_from_euler(0, 0, yaw))

    def navigate_to_node(self, node_id, start_orientation, target_orientation=None):
        """Navega a un nodo específico del grafo"""
        try:
            if node_id not in self.optimized_graph['nodes']:
                rospy.logerr(f"Nodo {node_id} no existe en el grafo")
                return False
            
            # Obtener posición del nodo en metros
            node_data = self.optimized_graph['nodes'][node_id]
            x_m, y_m = (node_data['x_m'], node_data['y_m'])
            
            # Verificar distancia al nodo
            if self.current_pose:
                dist = math.hypot(
                    x_m - self.current_pose.position.x,
                    y_m - self.current_pose.position.y
                )
                if dist < 0.1:  # Ya está en el nodo
                    rospy.loginfo(f"Ya está en el nodo {node_id}")
                    return True
            
            # Crear objetivo de navegación
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position = Point(x_m, y_m, 0)
            goal.target_pose.pose.orientation = self.calculate_orientation(start_orientation, target_orientation)
            
            # Guardar posición objetivo para cálculos de orientación
            self.goal_position = goal.target_pose.pose.position
            
            # Enviar objetivo
            rospy.loginfo(f"Navegando hacia nodo {node_id} en ({x_m:.2f}, {y_m:.2f})")
            goal.target_pose.pose.position = Point(x_m - self.map_origin_x, y_m - self.map_origin_y, 0)
            self.move_base_client.send_goal(goal)
            
            # Esperar resultado
            wait = self.move_base_client.wait_for_result(timeout=self.move_base_timeout)
            
            if not wait:
                rospy.logwarn(f"Tiempo de espera agotado para nodo {node_id}")
                self.move_base_client.cancel_goal()
                return False
            
            result = self.move_base_client.get_result()
            
            rospy.loginfo(f"Posición robot: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f})")

            # Verificar si llegó realmente cerca del nodo
            if result and self.current_pose:
                final_dist = math.hypot(
                    x_m - self.current_pose.position.x,
                    y_m - self.current_pose.position.y
                )
                if final_dist > self.max_node_distance:
                    rospy.logwarn(f"Llegó a {final_dist:.2f}m del nodo {node_id} (más allá del umbral)")
                    return False
            
            return result
        except Exception as e:
            rospy.logerr(f"Error en navigate_to_node: {str(e)}")

    def follow_path(self, path, start_orientation, target_orientation=None):
        """Versión optimizada con seguimiento de progreso"""
        if not path:
            rospy.logwarn("Ruta vacía recibida")
            return False
        
        total_nodes = len(path)
        
        for i, node_id in enumerate(path):
            rospy.loginfo(f"Progreso: {i+1}/{total_nodes} - Nodo actual: {node_id}")
            
            final_orientation = target_orientation if (i == len(path)-1) else None
            
            if not self.navigate_to_node(node_id, start_orientation, final_orientation):
                rospy.logerr(f"Fallo al alcanzar nodo {node_id}. Replanificando...")
                
                # Intentar replanificar desde la posición actual
                if self.current_node:
                    new_path_info = self.planner.get_detailed_path(self.current_node, path[-1])
                    if new_path_info and new_path_info['waypoints']:
                        rospy.loginfo(f"Nueva ruta: {' -> '.join(new_path_info['waypoints'])}")
                        return self.follow_path(new_path_info['waypoints'], start_orientation, target_orientation)
                
                rospy.logerr("No se pudo replanificar la ruta. Abortando.")
                return False
            
            # Pequeña pausa entre nodos para estabilización
            rospy.sleep(0.5)
        
        rospy.loginfo("¡Ruta completada con éxito!")
        return True

    def user_interface(self):
        """Interfaz de usuario mejorada con verificaciones"""
        if not self.graph_loaded:
            rospy.logerr("No se puede iniciar interfaz: grafo no cargado")
            return
            
        if not self.move_base_ready:
            rospy.logerr("No se puede iniciar interfaz: move_base no disponible")
            return
            
        if self.current_pose is None:
            rospy.logerr("No se puede iniciar interfaz: sin localización")
            return

        rospy.loginfo("\n\n=== INTERFAZ DE NAVEGACIÓN TOPOLÓGICA ===")
        rospy.loginfo("Sistema listo para recibir comandos\n")
        
        try:
            nodes = sorted(self.optimized_graph['nodes'].keys())
            
            while not rospy.is_shutdown():
                try:
                    # Mostrar información del estado actual
                    rospy.loginfo("\n" + "="*50)
                    rospy.loginfo(f"POSICIÓN ACTUAL: Nodo {self.current_node}")
                    rospy.loginfo(f"Coordenadas: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f})")
                    
                    rospy.loginfo("\nMENU PRINCIPAL:")
                    rospy.loginfo("1. Navegar a nodo específico")
                    rospy.loginfo("2. Mostrar grafo completo")
                    rospy.loginfo("3. Mostrar lista de nodos")
                    rospy.loginfo("4. Obtener pose actual")
                    rospy.loginfo("5. Obtener nodo actual")
                    rospy.loginfo("q. Salir")
                    rospy.loginfo("\nSeleccione una opción: ")
                    
                    option = input().strip()
                    
                    if option == 'q':
                        rospy.loginfo("Saliendo por solicitud del usuario")
                        rospy.signal_shutdown("Usuario solicitó salir")
                        break
                        
                    elif option == '1':
                        self.handle_navigation_option(nodes)
                    elif option == '2':
                        self.show_full_graph()
                    elif option == '3':
                        self.show_node_list()
                    elif option == '4':
                        self.get_pose()
                    elif option == '5':
                        self.get_node()
                    else:
                        rospy.logwarn("Opción no válida")
                        
                except Exception as e:
                    rospy.logerr(f"Error en interfaz de usuario: {str(e)}")
                    rospy.sleep(1)  # Pequeña pausa para evitar bucles rápidos de error
                    
        except KeyboardInterrupt:
            rospy.loginfo("Interfaz terminada por interrupción de teclado")
        finally:
            rospy.loginfo("Cerrando interfaz de usuario")

    def handle_navigation_option(self, nodes):
        """Maneja la opción de navegación a nodo específico"""
        rospy.loginfo("\nNODOS DISPONIBLES:")
        for i, node in enumerate(nodes):
            rospy.loginfo(f"{i+1}. {node}")
        
        rospy.loginfo("\nIngrese el nombre del nodo destino o 'b' para volver: ")
        target_node = input().strip()
        
        if target_node.lower() == 'b':
            return
            
        if target_node not in nodes:
            rospy.logwarn(f"Nodo {target_node} no válido")
            return
            
        # Solicitar orientaciones
        rospy.loginfo("\nOrientación inicial (vector dirección):")
        rospy.loginfo("Ejemplo: Para mirar hacia el este ingrese '1 0'")
        rospy.loginfo("Ingrese x y separados por espacio: ")
        start_ori = input().strip().split()
        
        try:
            start_orientation = [float(start_ori[0]), float(start_ori[1])]
            rospy.loginfo("\nOrientación final (opcional, presione Enter para omitir):")
            rospy.loginfo("Ingrese x y separados por espacio: ")
            target_ori = input().strip().split()
            target_orientation = [float(target_ori[0]), float(target_ori[1])] if target_ori else None
        except Exception as e:
            rospy.logerr(f"Error en orientación: {str(e)}")
            return
            
        # Planificar y seguir ruta
        path_info = self.planner.get_detailed_path(self.current_node, target_node)
        
        try:
            self.planner.visualize_path(path_info, show_full_graph=False)
        except Exception as e:
            rospy.logwarn(f"No se pudo visualizar ruta: {str(e)}")
        
        rospy.loginfo(f"\nRuta planificada: {' -> '.join(path_info['waypoints'])}")
        rospy.loginfo("¿Iniciar navegación? (s/n): ")
        confirm = input().strip().lower()
        
        if confirm == 's':
            success = self.follow_path(path_info['waypoints'], start_orientation, target_orientation)
            if success:
                rospy.loginfo("¡Destino alcanzado con éxito!")
            else:
                rospy.logerr("Fallo en la navegación")

    def show_full_graph(self):
        """Muestra el grafo completo"""
        try:
            self.graph_io.visualize_graph(self.optimized_graph)
            rospy.loginfo("Grafo mostrado en ventana emergente")
        except Exception as e:
            rospy.logerr(f"Error al visualizar grafo: {str(e)}")

    def show_node_list(self):
        """Muestra lista detallada de nodos"""
        rospy.loginfo("\nLISTA COMPLETA DE NODOS:")
        rospy.loginfo("{:<10} {:<15} {:<15} {:<15}".format("Nodo", "X (px)", "Y (px)", "Tipo"))
        for node_id, attrs in self.optimized_graph['nodes'].items():
            node_type = "Puerta" if node_id.startswith('D') else "Región"
            rospy.loginfo("{:<10} {:<15} {:<15} {:<15}".format(
                node_id, 
                str(attrs['x_m']), 
                str(attrs['y_m']), 
                node_type
            ))
    def get_pose(self):
        """Devuelve la pose actual del robot"""
        rospy.loginfo(f"Coordenadas robot: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f})")
    def get_node(self):
        """Devuelve el nodo actual del robot"""
        rospy.loginfo(f"Posición actual: Nodo {self.current_node}")

if __name__ == '__main__':
    try:
        navigator = TopologicalNavigation()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navegación topológica finalizada")
