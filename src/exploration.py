#!/usr/bin/env python

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Quaternion
import numpy as np
import time
import os
import math
from tf.transformations import quaternion_from_euler

# Variables globales
map_data = None
invalid_goals = set()  # Conjunto de objetivos inválidos
current_pose = None    # Pose actual del robot
start_time = None      # Tiempo de inicio de la exploración
explored_percentage = 0.0  # Porcentaje de exploración
iteration_count = 0    # Contador de iteraciones
consecutive_failures = 0  # Contador de fallos consecutivos
map_name = "escenario_4"  # Nombre del mapa (MODIFICAR PRIMERO!!!!!!)
previous_position = None  # Posición anterior del robot
stuck_count = 0  # Contador de veces que el robot no se mueve

# Parámetros de parada
MAX_ITERATIONS = 100  # Límite de iteraciones
MAX_CONSECUTIVE_FAILURES = 10  # Límite de fallos consecutivos
MAX_STUCK_COUNT = 10  # Límite de veces que el robot no se mueve

def map_callback(map_msg):
    """
    Callback para recibir el mapa generado por SLAM.
    """
    global map_data
    map_data = map_msg

def odom_callback(odom_msg):
    """
    Callback para recibir la odometría (pose actual del robot).
    """
    global current_pose
    current_pose = odom_msg.pose.pose

def calculate_explored_percentage(map_data):
    """
    Calcula el porcentaje de celdas exploradas respecto al total de celdas libres.
    """
    if map_data is None:
        return 0.0

    width = map_data.info.width
    height = map_data.info.height
    data = np.array(map_data.data).reshape((height, width))

    # Contar celdas libres y exploradas
    free_cells = np.sum(data == 0)  # Celdas libres
    explored_cells = np.sum(data != -1)  # Celdas exploradas (libres u ocupadas)

    if free_cells == 0:
        return 0.0

    return (explored_cells / free_cells) * 100.0

def is_goal_valid(goal_x, goal_y, map_data):
    """
    Verifica si un objetivo es viable (no está demasiado cerca de una pared).
    """
    width = map_data.info.width
    height = map_data.info.height
    data = np.array(map_data.data).reshape((height, width))

    # Verificar celdas adyacentes al objetivo
    for dy in range(-2, 3):  # Rango de 2 celdas alrededor del objetivo
        for dx in range(-2, 3):
            x = goal_x + dx
            y = goal_y + dy
            if 0 <= x < width and 0 <= y < height:
                if data[y, x] == 100:  # Celda ocupada (pared)
                    return False
    return True

def find_frontier(map_data):
    """
    Encuentra las celdas en la frontera entre lo conocido y lo desconocido.
    """
    frontiers = []
    width = map_data.info.width
    height = map_data.info.height
    data = np.array(map_data.data).reshape((height, width))

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if data[y, x] == 0:  # Celda libre
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if data[y + dy, x + dx] == -1:  # Celda desconocida adyacente
                        frontiers.append((x, y))
                        break
    return frontiers

def calculate_orientation(current_x, current_y, goal_x, goal_y):
    """
    Calcula la orientación óptima para moverse desde la posición actual hasta el objetivo.
    """
    dx = goal_x - current_x
    dy = goal_y - current_y
    yaw = math.atan2(dy, dx)  # Ángulo en radianes
    quat = quaternion_from_euler(0, 0, yaw)  # Convertir a cuaternión
    return Quaternion(*quat)

def save_map():
    """
    Guarda el mapa generado en la carpeta 'results'.
    """
    if not os.path.exists('results'):
        os.makedirs('results')

    # Guardar el mapa utilizando map_server
    os.system('rosrun map_server map_saver -f {}'.format(os.path.join('results', map_name)))

    print('Mapa guardado en results/{}.pgm'.format(map_name))

def save_metrics(execution_time, explored_percentage):
    """
    Guarda las métricas en un archivo de texto.
    """
    with open(f'results/metrics_{map_name}.txt', 'w') as f:
        f.write('Tiempo de ejecución: {:.2f} segundos\n'.format(execution_time))
        f.write('Porcentaje de exploración: {:.2f}%\n'.format(explored_percentage))
        f.write('Objetivos inválidos: {}\n'.format(len(invalid_goals)))
        f.write('Puntos no alcanzados: {}\n'.format(list(invalid_goals)))

    print(f'Mapa guardado en results/metrics_{map_name}.txt')

def select_and_publish_goal():
    """
    Selecciona un objetivo en la frontera y lo publica en move_base.
    """
    global map_data, invalid_goals, current_pose, explored_percentage, iteration_count, consecutive_failures, previous_position, stuck_count

    if map_data is not None and current_pose is not None:
        frontiers = find_frontier(map_data)
        if not frontiers:
            print('No se encontraron más fronteras. Exploración completada.')
            save_map()
            execution_time = time.time() - start_time
            save_metrics(execution_time, explored_percentage)
            rospy.signal_shutdown('Exploración completada')
            return

        print('Encontradas {} fronteras'.format(len(frontiers)))

        # Seleccionar un objetivo válido
        goal_found = False
        for _ in range(len(frontiers)):  # Intentar con todas las fronteras
            goal_x, goal_y = frontiers[np.random.randint(0, len(frontiers))]
            if (goal_x, goal_y) not in invalid_goals and is_goal_valid(goal_x, goal_y, map_data):
                goal_found = True
                break

        if goal_found:
            # Calcular la posición objetivo en coordenadas del mundo
            goal_world_x = goal_x * map_data.info.resolution + map_data.info.origin.position.x
            goal_world_y = goal_y * map_data.info.resolution + map_data.info.origin.position.y

            # Calcular la orientación óptima
            orientation = calculate_orientation(current_pose.position.x, current_pose.position.y, goal_world_x, goal_world_y)

            # Imprimir la pose actual y la pose objetivo
            print('Pose actual del robot:')
            print('  Posición: x={:.2f}, y={:.2f}'.format(current_pose.position.x, current_pose.position.y))
            print('  Orientación: w={:.2f}'.format(current_pose.orientation.w))
            print('Pose objetivo:')
            print('  Posición: x={:.2f}, y={:.2f}'.format(goal_world_x, goal_world_y))
            print('  Orientación: x={:.2f}, y={:.2f}, z={:.2f}, w={:.2f}'.format(
                orientation.x, orientation.y, orientation.z, orientation.w))

            print('Destino elegido. Navegando hasta el punto...')

            goal_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
            goal_client.wait_for_server()
            
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = goal_world_x
            goal.target_pose.pose.position.y = goal_world_y
            goal.target_pose.pose.orientation = orientation  # Orientación dinámica

            goal_client.send_goal(goal)
            wait = goal_client.wait_for_result(timeout=rospy.Duration(30))  # Esperar 30 segundos

            if not wait:
                print('No se pudo alcanzar el objetivo. Marcando como inválido.')
                invalid_goals.add((goal_x, goal_y))  # Marcar objetivo como inválido
                consecutive_failures += 1  # Incrementar fallos consecutivos
                goal_client.cancel_goal()  # Cancelar el objetivo actual
            else:
                # Verificar si el robot se ha movido
                if previous_position is not None:
                    distance_moved = math.sqrt(
                        (current_pose.position.x - previous_position.x) ** 2 +
                        (current_pose.position.y - previous_position.y) ** 2
                    )
                    if distance_moved < 0.1:  # Umbral de movimiento (10 cm)
                        print('El robot no se movió. Marcando objetivo como inválido.')
                        invalid_goals.add((goal_x, goal_y))  # Marcar objetivo como inválido
                        stuck_count += 1
                    else:
                        print('Punto alcanzado!')
                        stuck_count = 0
                        # Reiniciar fallos consecutivos si el objetivo se alcanzó
                        consecutive_failures = 0
            
            # Actualizar la posición anterior
            previous_position = current_pose.position

            # Actualizar el porcentaje de exploración
            explored_percentage = calculate_explored_percentage(map_data)
            print('Porcentaje de exploración: {:.2f}%'.format(explored_percentage))
            print('Robot en posición: x={:.2f}, y={:.2f}'.format(current_pose.position.x, current_pose.position.y))

            # Incrementar el contador de iteraciones
            iteration_count += 1

            # Verificar criterio de parada
            if (iteration_count >= MAX_ITERATIONS or
                consecutive_failures >= MAX_CONSECUTIVE_FAILURES or
                stuck_count >= MAX_STUCK_COUNT):
                print('Criterio de parada alcanzado.')
                save_map()
                execution_time = time.time() - start_time
                save_metrics(execution_time, explored_percentage)
                rospy.signal_shutdown('Exploración completada')
        else:
            print('No se encontraron objetivos válidos. Exploración completada.')
            save_map()
            execution_time = time.time() - start_time
            save_metrics(execution_time, explored_percentage)
            rospy.signal_shutdown('Exploración completada')
    
    map_data = None

if __name__ == '__main__':
    try:
        rospy.init_node('exploration', anonymous=True)
        map_data = None
        current_pose = None
        start_time = time.time()  # Iniciar el tiempo de exploración

        # Suscribirse al mapa y a la odometría
        map_subscriber = rospy.Subscriber('/map', OccupancyGrid, map_callback)
        odom_subscriber = rospy.Subscriber('/odom', Odometry, odom_callback)
        
        rate = rospy.Rate(1)  # Frecuencia de 1 Hz

        while not rospy.is_shutdown():
            select_and_publish_goal()
            rate.sleep()

    except rospy.ROSInterruptException:
        rospy.loginfo("Exploration finished.")