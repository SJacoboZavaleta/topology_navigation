#!/usr/bin/env python3

"""
Libreria para la creación de mapas topológicos
Curso: Robotica Movil
Autor: Sergio Jacobo Zavaleta
"""

from math import hypot, sqrt, atan2, degrees
from collections import defaultdict
from heapq import heappush, heappop
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
import networkx as nx

import json
import gzip
import pickle
from pathlib import Path

from itertools import combinations

import cv2
import yaml
from skimage import morphology
from shapely.geometry import Point, Polygon
from scipy.spatial import distance
import pandas as pd
from scipy.spatial import cKDTree, KDTree
from skimage.draw import line

from pathlib import Path
from datetime import datetime

class TopologicalMapProcessor:
    class Config:
        """Clase para almacenar todos los parámetros configurables"""
        def __init__(self):
            # Parámetros de preprocesamiento del mapa
            self.robot_diametro = 0.3         # Diámetro del robot en metros
            self.dilate_robot_iterations = 1   # Iteraciones para dilatación del robot
            """
            Efecto: Agregar dilatación para considerar el tamaño del robot
            robot_diametro: 0.3
            dilate_robot_iterations: 1
            """
            self.morph_open_kernel = (15, 15)  # Tamaño del kernel para opening morfológico
            """
            Efecto: Elimina ruido y pequeños obstáculos del mapa
            - Aumentar: Elimina más ruido pero puede perder detalles finos
            - Disminuir: Preserva más detalles pero mantiene más ruido
            Valor típico: (5,5) a (15,15)
            """
            self.erode_kernel = (3, 3)        # Tamaño del kernel para erosión
            self.erode_iterations = 1         # Iteraciones para erosión
            """
            Efecto combinado: Reduce el área de regiones navegables
            - Aumentar kernel/iteraciones: Crea pasillos más estrechos, elimina más ruido
            - Disminuir: Mantiene áreas más amplias pero conserva más artefactos
            Valor típico: Kernel (3,3) a (5,5), Iteraciones 1-3
            """
            # Parámetros de segmentación
            self.watershed_threshold = 0.6   # Umbral para watershed
            """
            Efecto: Controla el tamaño de las regiones segmentadas
            - Aumentar (→1.0): Crea menos y más grandes regiones
            - Disminuir (→0.0): Produce más regiones pequeñas
            Valor óptimo: 0.4-0.6 para la mayoría de mapas
            """
            self.dilate_iterations = 2        # Iteraciones para dilatación
            """
            Efecto: Expande las áreas para la segmentación inicial
            - Aumentar: Mejora la cobertura pero puede unir regiones distintas
            - Disminuir: Preserva separaciones pero puede dejar áreas sin cubrir
            Valor típico: 1-3 iteraciones
            """
            # Parámetros de esqueletización
            self.skeleton_erode_kernel = (13, 13)  # Kernel para erosión antes de esqueletización
            """
            Efecto: Grosor inicial para la extracción del esqueleto
            - Aumentar: Produce esqueletos más simples pero puede perder detalles
            - Disminuir: Conserva más detalles pero puede introducir ramas espurias
            Valor típico: (5,5) a (15,15)
            """
            # Parámetros de detección de nodos
            self.curvature_threshold = 0.7    # Umbral para detección de curvatura
            """
            Efecto: Sensibilidad para detectar esquinas/cambios de dirección
            - Aumentar (→1.0): Solo detecta curvas muy pronunciadas
            - Disminuir (→0.0): Detecta hasta ligeros cambios de dirección
            Valor óptimo: 0.3-0.7
            """
            self.strategic_node_step = 30     # Paso para nodos estratégicos en caminos largos
            """
            Efecto: Densidad de nodos en caminos largos
            - Aumentar: Menos nodos, grafo más simple
            - Disminuir: Más nodos, mayor precisión pero más complejidad
            Valor típico: 10-30 píxeles
            """
            # Parámetros de conexión entre nodos
            self.max_gap = 3                  # Huecos máximos permitidos en conexiones normales
            self.max_gap_door = 30            # Huecos máximos para conexiones con puertas
            """
            Efecto: Tolerancia a discontinuidades en el esqueleto
            - Aumentar: Conecta a través de mayores discontinuidades
            - Disminuir: Requiere conexiones más continuas
            Valores típicos:
            - Normal: 3-10 píxeles
            - Puertas: 15-30 píxeles
            """
            self.door_connection_threshold = 300  # Distancia máxima puerta-nodo
            """
            Efecto: Radio de conexión para nodos de puertas
            - Aumentar: Conecta puertas con nodos más lejanos
            - Disminuir: Restringe conexiones a nodos cercanos
            Valor típico: 100-300 píxeles
            """
            # Parámetros de visualización
            self.skeleton_node_size = 100     # Tamaño de nodos en visualización
            self.skeleton_line_width = 2      # Grosor de líneas en visualización
            self.graph_node_size_base = 30    # Tamaño base de nodos
            self.graph_node_size_factor = 20  # Factor de tamaño por grado
            self.door_node_size = 150         # Tamaño de nodos puerta
            """
            Estos parámetros solo afectan la visualización:
            - Valores mayores mejoran visibilidad en mapas grandes
            - Valores menores permiten mayor densidad de información
            Ajustar según necesidad de visualización, no afectan el procesamiento
            """
    
    def __init__(self, yaml_file, config=None):
        self.yaml_file = yaml_file
        self.config = config if config else self.Config()
        self.mapa = None
        self.mapa_binario = None
        self.skeleton = None
        self.coord_subnodos = []
        self.skeleton_graph = None
        self.regionSubNodos = []
        self.etiquetas = None
        self.puntosCentrales = []
        self.conexiones = []
        self.centroides = []
        self.MAPS_DIR = Path(__file__).parent.parent.parent / "maps"
        self.graph = {
            "meta": {
                "graph_type": "hierarchical_topological",
                "levels": ["high_level_regions", "low_level_waypoints"]
            },
            "regions": {},
            "doors": [],
            "connections": []
        }

    def load_and_process_map(self, path=None):
        """Carga y procesa el mapa inicial"""
        with open(self.yaml_file, 'r') as file:
            metadatos = yaml.safe_load(file)

        map_file = self.MAPS_DIR / metadatos["image"]
        self.mapa = cv2.imread(map_file, -1)
        self.res = metadatos["resolution"]
        self.oriX, self.oriY = metadatos["origin"][:2]
        
        # Crear mapa binario
        self.mapa_binario = np.where((self.mapa == 254), 255, 0).astype(np.uint8)

        # Encontrar el primer píxel blanco (esquina superior izquierda de la casa) ---
        # Buscar la primera fila y columna donde aparece un píxel blanco (255)
        filas, columnas = np.where(self.mapa_binario == 255)
        self.escenario_x_min_pix = np.min(columnas)  # Columna más a la izquierda de la casa
        self.escenario_y_min_pix = np.min(filas)     # Fila más superior de la casa

        print(f"Primer píxel blanco (esquina casa) en píxeles: ({self.escenario_x_min_pix}, {self.escenario_y_min_pix})")

        # Visualización del mapa binario con la esquina de la casa segun Gazebo
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(self.mapa_binario, cmap='gray')
        plt.title("Mapa binario (original)")
        plt.plot(self.escenario_x_min_pix, self.escenario_y_min_pix, 'ro')  # Marcar esquina casa

        plt.subplot(1, 2, 2)
        # Mapa con ejes en metros (ejemplo)
        plt.imshow(self.mapa_binario, cmap='gray', extent=[
            -self.escenario_x_min_pix * self.res,
            (self.mapa_binario.shape[1] - self.escenario_x_min_pix) * self.res,
            -(self.mapa_binario.shape[0] - self.escenario_y_min_pix) * self.res,
            self.escenario_y_min_pix * self.res
        ])
        plt.title("Mapa con ejes en metros (relativo al mapa en Gazebo)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.grid(True)
        plt.show()


        self.mapa_binario = cv2.morphologyEx(self.mapa_binario, cv2.MORPH_OPEN, 
                                             np.ones((self.config.morph_open_kernel), np.uint8)) ####### parametro
        self.mapa_binario = cv2.erode(self.mapa_binario, np.ones(self.config.erode_kernel, np.uint8), iterations=self.config.erode_iterations) ####### parametro

        
        plt.figure(figsize=(6, 6))
        plt.imshow(self.mapa_binario, cmap='gray')
        plt.title("Mapa Binario", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        if path:
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

    def pix_to_metros(self, x_pix, y_pix):
        """Convierte coordenadas de píxeles a metros segun Gazebo"""
        # Encontrado el pixel del origen físico del escenario dentro del mapa correspondiente al mapa de Gazebo.
        # Por ejemplo, el escenario 4, su origen coincide o es cercano a (0,0) en la esquina superior izquierda
        # Resolución del mapa: 0.05 metros por pixel
        x_m = (x_pix - self.escenario_x_min_pix) * self.res
        y_m = -(y_pix - self.escenario_y_min_pix) * self.res
        return x_m, y_m
    
    def segment_map(self, path=None):
        """Erosión inicial para considerar el tamaño del robot"""
        diametro_robot_pix = int(self.config.robot_diametro / self.res) ####### parametro
        # Kernel de erosión (tamaño basado en el robot)
        kernel_robot = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diametro_robot_pix, diametro_robot_pix))
        self.mapa_binario = cv2.erode(self.mapa_binario, kernel_robot, iterations=self.config.dilate_robot_iterations)####### parametro

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(self.mapa_binario, cmap='gray')
        plt.title("Mapa erosionado (seguro para robot)")
        plt.axis("off")
        
        """Segmenta el mapa usando watershed"""
        fondo = cv2.dilate(self.mapa_binario, np.ones((3,3), np.uint8), iterations=self.config.dilate_iterations)####### parametro
        dist_transform = cv2.distanceTransform(self.mapa_binario, cv2.DIST_L2, 5)
        _, frente = cv2.threshold(dist_transform, self.config.watershed_threshold * dist_transform.max(), 255, 0)####### parametro
        desconocido = cv2.subtract(fondo, np.uint8(frente))
        
        _, etiquetas = cv2.connectedComponents(np.uint8(frente))
        self.etiquetas = etiquetas + 1
        self.etiquetas[desconocido == 255] = 0
        self.etiquetas = cv2.watershed(cv2.cvtColor(self.mapa, cv2.COLOR_GRAY2BGR), self.etiquetas)
        
        # Extraer mapa segmentado coloreado
        # colormap = np.random.randint(0, 255, (self.etiquetas.max() + 1, 3), dtype=np.uint8)
        fixed_colormap = np.array([
            [230, 25, 75],    # Rojo vibrante
            [60, 180, 75],     # Verde esmeralda
            [255, 225, 25],    # Amarillo brillante
            [0, 130, 200],     # Azul cielo
            [245, 130, 48],    # Naranja
            [145, 30, 180],    # Púrpura
            [70, 240, 240],    # Cyan
            [240, 50, 230],    # Rosa
            [210, 245, 60],    # Lima
            [250, 190, 190],   # Rosa claro
            [0, 128, 128],     # Verde azulado
            [230, 190, 255],   # Lavanda
            [170, 110, 40],    # Marrón
            [255, 250, 200],   # Beige claro
            [128, 0, 0],       # Rojo oscuro
            [170, 255, 195],   # Verde menta
            [128, 128, 0],     # Oliva
            [255, 215, 180],   # Melocotón
            [0, 0, 128],       # Azul marino
            [128, 128, 128],   # Gris
            ], dtype=np.uint8)
        
        colormap = np.vstack([
            [[255, 255, 255]],  # Fondo blanco (etiqueta 1)
            fixed_colormap       # Colores fijos para las regiones (etiquetas 2+)
        ])
        colormap[1] = [255, 255, 255]
        self.etiquetas_coloreadas = colormap[self.etiquetas]
        self.etiquetas_coloreadas = cv2.convertScaleAbs(self.etiquetas_coloreadas)

        # Calcular centroides
        self.centroides = []
        for etiqueta in range(2, self.etiquetas.max() + 1):
            mascara = (self.etiquetas == etiqueta).astype(np.uint8)
            momentos = cv2.moments(mascara)
            if momentos["m00"] != 0:
                cX = int(momentos["m10"] / momentos["m00"])
                cY = int(momentos["m01"] / momentos["m00"])
                self.centroides.append((cX, cY))

        
        plt.figure(figsize=(6, 6))
        plt.imshow(self.etiquetas_coloreadas)
        plt.title("Mapa Segmentado", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        if path:
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

    def extract_topological_info(self, path=None):
        """Extrae información topológica (puertas y conexiones)"""
        vecinos = []
        pxVecinos = []
        indices = np.where(self.etiquetas == -1)
        indices_as_tuples = np.transpose(indices)
        
        for index in indices_as_tuples:
            if all([0 < index[0] < self.etiquetas.shape[0]-1, 
                   0 < index[1] < self.etiquetas.shape[1]-1]):
                idxVecino = np.indices((3,3)) + np.array(index)[:, np.newaxis, np.newaxis] - 1
                idxVecino = idxVecino.reshape(2, -1).T
                neighbors = self.etiquetas[tuple(idxVecino.T)]
                
                if not np.any(neighbors == 1):
                    vecino = np.unique(neighbors[neighbors > 0])
                    if len(vecino) == 2:
                        vecinos.append(vecino)
                        pxVecinos.append(index)
        
        separaciones = np.zeros_like(self.etiquetas, dtype=np.uint8)
        for pixel in pxVecinos:
            separaciones[pixel[0], pixel[1]] = 255
            
        num_labels, labeled_image, stats, centroids = cv2.connectedComponentsWithStats(separaciones, 8)
        
        self.puntosCentrales = []
        self.conexiones = []
        for label in range(1, num_labels):
            pixels = np.where(labeled_image == label)
            if len(pixels[0]) > 0:
                pxCentral = (pixels[0][len(pixels[0])//2], pixels[1][len(pixels[0])//2])
                self.puntosCentrales.append(pxCentral)
                
                for i, px in enumerate(pxVecinos):
                    if (px == np.array(pxCentral)).all():
                        self.conexiones.append(vecinos[i])
                        break

        
        plt.figure(figsize=(6, 6))
        plt.imshow(self.etiquetas_coloreadas, cmap='tab20')
        plt.title("Mapa Topológico: Puertas y Conexiones", fontsize=12)
        plt.axis("off")
        
        mapa_centros = {i + 2: self.centroides[i] for i in range(len(self.centroides))}
        for idx, (punto_sep, etiquetas) in enumerate(zip(self.puntosCentrales, self.conexiones)):
            etiqueta1, etiqueta2 = etiquetas
            if etiqueta1 in mapa_centros and etiqueta2 in mapa_centros:
                centro1 = mapa_centros[etiqueta1]
                centro2 = mapa_centros[etiqueta2]
                plt.plot([punto_sep[1], centro1[0]], [punto_sep[0], centro1[1]], 'b-', linewidth=1)
                plt.plot([punto_sep[1], centro2[0]], [punto_sep[0], centro2[1]], 'b-', linewidth=1)
        
        for punto in self.centroides:
            plt.scatter(punto[0], punto[1], c='red', s=50)
        
        for punto in self.puntosCentrales:
            plt.scatter(punto[1], punto[0], c='blue', s=50)
        
        plt.tight_layout()
        if path:
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

    def extract_voronoi_nodes(self, path=None):
        """Extrae nodos de traversabilidad basados en el esqueleto morfológico"""
        # 1. Esqueletización
        mapa_erosionado = cv2.morphologyEx(self.mapa_binario, cv2.MORPH_ERODE, np.ones(self.config.skeleton_erode_kernel, np.uint8))#### parametro
        self.skeleton = morphology.medial_axis(mapa_erosionado)
        
        # 2. Detección de nodos críticos
        self._find_critical_nodes()
        
        # 3. Añadir nodos estratégicos en caminos largos
        self._add_strategic_nodes()
        
        # 4. Construir grafo robusto del esqueleto
        self._build_robust_skeleton_graph()
        
        # 5. Asignar regiones a los nodos
        self._assign_regions_to_nodes()
        
        # 6. Visualización del grafo del esqueleto
        self._visualize_skeleton_graph(path)

    def _find_critical_nodes(self):
        """Detección mejorada de nodos críticos basada en el esqueleto"""
        self.coord_subnodos = []
        # Original skeleton pixels
        # skeleton_pixels = list(zip(*np.where(self.skeleton)))
        # Ordenar las coordenadas para consistencia (por fila, luego por columna)
        skeleton_pixels = sorted(zip(*np.where(self.skeleton)), key=lambda k: (k[0], k[1]))

        for y, x in skeleton_pixels:
            neighborhood = self.skeleton[max(y-1,0):min(y+2, self.skeleton.shape[0]), 
                             max(x-1,0):min(x+2, self.skeleton.shape[1])]
            connections = neighborhood.sum() - 1
            
            if connections != 2 or self._is_curvature_point((y, x)):
                self.coord_subnodos.append((y, x))

    def _is_curvature_point(self, point):
        """Detecta puntos de curvatura en el esqueleto"""
        y, x = point
        directions = []
        
        for dy, dx in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.skeleton.shape[0] and 0 <= nx < self.skeleton.shape[1]:
                if self.skeleton[ny, nx]:
                    directions.append((dy, dx))
        
        if len(directions) == 2:
            # Ordenar direcciones para consistencia (ej: [(0,1), (1,0)] -> [(0,1), (1,0)])
            directions = sorted(directions, key=lambda k: (k[0], k[1]))
            dot_product = directions[0][0]*directions[1][0] + directions[0][1]*directions[1][1]
            return abs(dot_product) < self.config.curvature_threshold #### parametro
        return False

    def _add_strategic_nodes(self):#### parametro
        """Añade nodos intermedios en caminos largos"""
        step_size = self.config.strategic_node_step

        if not self.coord_subnodos:
            return
        
        # 1. Ordenar los píxeles del esqueleto para consistencia
        skeleton_pixels = sorted(zip(*np.where(self.skeleton)), key=lambda k: (k[0], k[1]))
        
        # 2. Convertir a array numpy para KDTree
        existing_nodes = np.array(self.coord_subnodos)
        tree = KDTree(existing_nodes)
        
        new_nodes = []
        # 3. Usar distancia mínima por celda para evitar duplicados
        min_distances = np.full(self.skeleton.shape, np.inf)
        
        for y, x in skeleton_pixels:
            dist, _ = tree.query([(y, x)], k=1)
            current_dist = dist[0]
            
            # 4. Verificar si es punto válido y con distancia suficiente
            if current_dist > step_size:
                neighborhood = self.skeleton[max(y-1,0):min(y+2, self.skeleton.shape[0]), 
                                max(x-1,0):min(x+2, self.skeleton.shape[1])]
                if neighborhood.sum() > 1 and current_dist < min_distances[y, x]:
                    min_distances[y, x] = current_dist
                    new_nodes.append((y, x))
        
        # 5. Añadir nuevos nodos ordenados
        self.coord_subnodos.extend(sorted(new_nodes, key=lambda k: (k[0], k[1])))


    def _build_robust_skeleton_graph(self):
        """Construye un grafo robusto basado en el esqueleto morfológico"""
        self.skeleton_graph = nx.Graph()
        
        # 1. Añadir nodos ordenados
        nodes = sorted(enumerate(self.coord_subnodos), key=lambda x: (x[1][0], x[1][1]))
        for i, (y, x) in nodes:
            self.skeleton_graph.add_node(i, pos=(y, x))
        
        # 2. Mapa de píxeles accesible (forma correcta)
        skeleton_matrix = np.zeros_like(self.skeleton, dtype=bool)
        y_idx, x_idx = np.where(self.skeleton)
        skeleton_matrix[y_idx, x_idx] = True
        
        # 3. Conectar nodos en orden determinista
        node_indices = sorted(self.skeleton_graph.nodes())
        for i, j in combinations(node_indices, 2):
            pos_i = self.coord_subnodos[i]
            pos_j = self.coord_subnodos[j]
            
            # 4. Conexión optimizada y determinista
            if self._are_nodes_connected_by_skeleton(pos_i, pos_j, skeleton_matrix):
                self.skeleton_graph.add_edge(i, j)

    def _are_nodes_connected_by_skeleton(self, pos1, pos2, skeleton_matrix):
        """Prueba de conexión mejorada"""
        y1, x1 = int(round(pos1[0])), int(round(pos1[1]))
        y2, x2 = int(round(pos2[0])), int(round(pos2[1]))
        
        # Bresenham line algorithm modificado
        line_coords = list(zip(*line(y1, x1, y2, x2)))
        
        # Verificar continuidad con tolerancia
        gap_count = 0
        for y, x in line_coords:
            if 0 <= y < skeleton_matrix.shape[0] and 0 <= x < skeleton_matrix.shape[1]:
                if not skeleton_matrix[y, x]:
                    gap_count += 1
                    if gap_count > self.config.max_gap:
                        return False
                else:
                    gap_count = 0
            else:
                return False
        
        return len(line_coords) > 0  # Conexión válida

    def _assign_regions_to_nodes(self):
        """Asigna regiones a los nodos del esqueleto
        Nueva versión: Asignación de regiones a nodos del esqueleto.
        Asignación directa de nodos en regiones válidas pero exteriores (1) a la región más cercana que origina la conexión pero que fueron aisladas por el esqueleto.

        Se mantiene parte de la logica anterior mediante el calculo de perimetros de las regiones y la asignación de etiquetas a los nodos, pero que se mejoro debido a su alto computo.
        """
        if self.etiquetas is None:
            return
        
        # 1. Precomputar centroides de región
        region_centroids = {}
        for i in range(2, self.etiquetas.max() + 1):
            y, x = np.where(self.etiquetas == i)
            region_centroids[i] = (np.mean(y), np.mean(x))
        
        # 2. KDTree para búsqueda rápida
        valid_regions = list(region_centroids.keys())
        kdtree = KDTree(list(region_centroids.values()))
        
        # 3. Asignación con doble estrategia: CV2 y proximidad
        self.regionSubNodos = []
        for y, x in self.coord_subnodos:
            region = self.etiquetas[y, x] if (0 <= y < self.etiquetas.shape[0] and 
                                            0 <= x < self.etiquetas.shape[1]) else -1
            
            # Caso 1: Región válida directa
            if region >= 2:
                self.regionSubNodos.append(region)
                continue
                
            # Caso 2: Asignación por proximidad (para nodos en región 1/-1)
            _, nearest_idx = kdtree.query((y, x))
            nearest_region = valid_regions[nearest_idx]
            
            # Verificación adicional de polígono
            if self._is_point_near_region(y, x, nearest_region):
                self.regionSubNodos.append(nearest_region)
            else:
                self.regionSubNodos.append(-1)  # Nodo en pared/espacio no navegable

    def _is_point_near_region(self, y, x, region_id, threshold=5):
        """Verifica si el punto está cerca del polígono de la región"""
        mask = (self.etiquetas == region_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convertir a enteros explícitamente
        point = (int(round(x)), int(round(y)))
        
        for cnt in contours:
            # Asegurar que el contorno tenga al menos 3 puntos
            if len(cnt) >= 3:
                dist = cv2.pointPolygonTest(cnt, point, True)  # Distancia con signo
                if abs(dist) <= threshold:
                    return True
        return False

    def _visualize_skeleton_graph(self, path=None):
        """Visualización del grafo del esqueleto"""
        node_size = self.config.skeleton_node_size
        line_width = self.config.skeleton_line_width

        plt.figure(figsize=(6, 6))
        overlay = cv2.addWeighted(cv2.cvtColor(self.mapa, cv2.COLOR_GRAY2BGR), 0.5, self.etiquetas_coloreadas, 0.5, 0)
        # Mostrar el mapa original
        plt.imshow(overlay)
        # Dibujar el esqueleto
        plt.imshow(self.skeleton, cmap='gray_r', alpha=0.3)
        
        # Dibujar nodos por grado
        for node in self.skeleton_graph.nodes():
            y, x = self.coord_subnodos[node]
            degree = self.skeleton_graph.degree[node]
            color = 'red' if degree == 1 else 'green' if degree == 2 else 'blue'
            plt.scatter(x, y, c=color, s=node_size, edgecolors='white', linewidths=1)
        
        # Dibujar conexiones
        for u, v in self.skeleton_graph.edges():
            y1, x1 = self.coord_subnodos[u]
            y2, x2 = self.coord_subnodos[v]
            plt.plot([x1, x2], [y1, y2], 'c-', linewidth=line_width, alpha=0.5)
        
        plt.title("Grafo del mapa de trasnversabilidad mejorado", fontsize=12)
        plt.axis('off')
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Endpoint (grado 1)',
                    markerfacecolor='red', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Intermediate (grado 2)',
                    markerfacecolor='green', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Junction (grado 3+)',
                    markerfacecolor='blue', markersize=10)
        ]

        plt.legend(handles=legend_elements, loc='upper right')
        plt.tight_layout()
        if path:
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()

    def build_topological_graph(self):
        """Construye el grafo topológico completo"""
        if not hasattr(self, 'skeleton_graph'):
            raise ValueError("Primero debe generar el esqueleto robusto")
        
        umbral_distancia_puerta = self.config.door_connection_threshold

        # 1. Procesar nodos por región
        region_nodes = defaultdict(list)
        for i, ((y, x), region) in enumerate(zip(self.coord_subnodos, self.regionSubNodos)):
            if region == -1:
                continue
                
            node_id = f"R{region}_N{i}"
            degree = self.skeleton_graph.degree(i)
            node_type = "endpoint" if degree == 1 else "intermediate" if degree == 2 else "junction"
            
            # Convertir coordenadas de pixeles a metros
            x_m, y_m = self.pix_to_metros(x, y)

            region_nodes[region].append({
                "id": node_id,
                "x": int(x),
                "y": int(y),
                "x_m": x_m,
                "y_m": y_m,
                "type": node_type,
                "region": int(region),
                "degree": degree,
                "color": "red" if degree == 1 else "green" if degree == 2 else "blue"
            })
        
        # 2. Procesar puertas
        door_nodes = []
        for i, ((y, x), (region_a, region_b)) in enumerate(zip(self.puntosCentrales, self.conexiones)):
            door_id = f"D{i}"

            # Convertir coordenadas de pixeles a metros
            x_m, y_m = self.pix_to_metros(x, y)

            door_nodes.append({
                "id": door_id,
                "x": int(x),
                "y": int(y),
                "x_m": x_m,
                "y_m": y_m,
                "type": "door",
                "connects": sorted([int(region_a), int(region_b)]),
                "color": "black"
            })
        
        # 3. Conexiones internas
        internal_connections = []
        for i, j in self.skeleton_graph.edges():
            region_i = self.regionSubNodos[i]
            region_j = self.regionSubNodos[j]
            if region_i == region_j and region_i != -1:
                internal_connections.append((f"R{region_i}_N{i}", f"R{region_j}_N{j}"))
        
        # 4. Conexiones puerta-nodo (mejorado)
        door_connections = []
        for door in door_nodes:
            door_pos = np.array([door["y"], door["x"]])
            
            for region in door["connects"]:
                closest_node = None
                min_dist = float('inf')
                
                for node_id in self.skeleton_graph.nodes():
                    if self.regionSubNodos[node_id] == region:
                        node_pos = np.array(self.coord_subnodos[node_id])
                        dist = np.linalg.norm(node_pos - door_pos)
                        
                        if dist < min_dist and dist < umbral_distancia_puerta:  #### parametro
                            if self._has_direct_connection_skeleton(door_pos, node_pos):
                                min_dist = dist
                                closest_node = node_id
                
                if closest_node is not None:
                    door_connections.append((door["id"], f"R{region}_N{closest_node}"))
        
        # Construir grafo final
        self.graph = {
            "meta": {
                "graph_type": "enhanced_topological",
                "levels": ["regions", "nodes"],
                "num_regions": len(region_nodes),
                "num_doors": len(door_nodes),
                "num_nodes": sum(len(nodes) for nodes in region_nodes.values()),
                "skeleton_nodes": len(self.coord_subnodos),
                "escenario_origen": [self.escenario_x_min_pix, self.escenario_y_min_pix],
            },
            "regions": dict(region_nodes),
            "doors": door_nodes,
            "connections": {
                "internal": internal_connections,
                "door": door_connections
            }
        }
        
        self._validate_graph_structure()

    def _has_direct_connection_skeleton(self, pos1, pos2):
        """Verifica conexión directa en el esqueleto con tolerancia a huecos
        >> Una discontinuidad en el esqueleto (línea blanca interrumpida).
        >> Píxeles negros/no navegables (valor 0) entre píxeles blancos/navegables (valor 1) del esqueleto.
        Como afecta max_gap?
        Es un parámetro de tolerancia que define:
        - El número máximo de píxeles negros consecutivos que se permiten en una conexión entre dos nodos.
        - Si se supera este valor, el algoritmo considera que no hay conexión válida.
        - Mayor tolerancia para puertas (regiones especiales)
        """
        max_gap = self.config.max_gap_door
        height, width = self.skeleton.shape[:2]
        
        # Convertir coordenadas a enteros con redondeo
        y1, x1 = map(round, pos1)
        y2, x2 = map(round, pos2)
        y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
        
        # Generar línea digital optimizada
        line_coords = list(zip(*line(y1, x1, y2, x2)))
        
        # Pre-allocar array de validación
        coords_array = np.array(line_coords)
        
        # Comprobación masiva de bordes
        valid_mask = ((coords_array[:,0] >= 0) & (coords_array[:,0] < height)) & ((coords_array[:,1] >= 0) & (coords_array[:,1] < width))
        
        if not np.all(valid_mask):
            return False
        
        # Vectorización de la comprobación del esqueleto
        skeleton_values = self.skeleton[coords_array[:,0], coords_array[:,1]]
        
        # Detección de huecos con tolerancia
        gap_count = 0
        for is_skeleton in skeleton_values:
            if not is_skeleton:
                gap_count += 1
                if gap_count > max_gap:
                    return False
            else:
                gap_count = 0
        
        return len(line_coords) > 0  # Asegurar que hay al menos un punto

    def _validate_graph_structure(self):
        """Valida la estructura del grafo"""
        # Verificar conexiones de puertas
        for door in self.graph["doors"]:
            connected = False
            for conn in self.graph["connections"]["door"]:
                if door["id"] in conn:
                    connected = True
                    break
            if not connected:
                print(f"Advertencia: Puerta {door['id']} sin conexiones")

    def visualize_graph(self, path=None):
        """Visualización del grafo topológico completo"""
        door_size = self.config.door_node_size

        plt.figure(figsize=(12, 10))
        
        if hasattr(self, 'etiquetas_coloreadas'):
            plt.imshow(self.etiquetas_coloreadas, alpha=0.3)
        
        # Dibujar nodos por región
        for region, nodes in self.graph["regions"].items():
            xs = [node["x"] for node in nodes]
            ys = [node["y"] for node in nodes]
            colors = [node["color"] for node in nodes]
            sizes = [self.config.graph_node_size_base + self.config.graph_node_size_factor * node["degree"] for node in nodes]
            # sizes = [30 + 20 * node["degree"] for node in nodes]
            
            plt.scatter(xs, ys, c=colors, s=sizes, label=f"Región {region}", 
                    edgecolors='white', linewidths=0.5, zorder=5)
        
        # Dibujar puertas
        for door in self.graph["doors"]:
            plt.scatter(door["x"], door["y"], c=door["color"], marker='s', 
                    s=door_size, label='Puerta', zorder=6, edgecolors='white')
        
        # Dibujar conexiones
        for a, b in self.graph["connections"]["internal"]:
            node_a = self._find_node(a)
            node_b = self._find_node(b)
            if node_a and node_b:
                plt.plot([node_a["x"], node_b["x"]], 
                        [node_a["y"], node_b["y"]],
                        color='magenta', linewidth=0.7, alpha=0.5, zorder=3)
        
        for a, b in self.graph["connections"]["door"]:
            node_a = self._find_node(a)
            node_b = self._find_node(b)
            if node_a and node_b:
                plt.plot([node_a["x"], node_b["x"]], 
                        [node_a["y"], node_b["y"]], 
                        'k--', linewidth=1.5, alpha=0.8, zorder=4)
        
        plt.title("Grafo Topológico", fontsize=12)
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Extremo (grado 1)',
                markerfacecolor='red', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Intermedio (grado 2)',
                markerfacecolor='green', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Bifurcación (grado 3+)',
                markerfacecolor='blue', markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label='Puerta',
                markerfacecolor='black', markersize=10)
        ]
        
        plt.legend(handles=legend_elements, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if path:
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()

    def _find_node(self, node_id):
        """Busca un nodo por ID"""
        if node_id.startswith('D'):
            for door in self.graph["doors"]:
                if door["id"] == node_id:
                    return door
        else:
            for region, nodes in self.graph["regions"].items():
                for node in nodes:
                    if node["id"] == node_id:
                        return node
        return None

    def generate_detailed_node_report(self, output_file=None):
        """Genera reporte detallado de nodos"""
        report_data = []
        
        # Nodos de regiones
        for region, nodes in self.graph["regions"].items():
            for node in nodes:
                connected_nodes = []
                for a, b in self.graph["connections"]["internal"]:
                    if a == node["id"]:
                        connected_nodes.append(b)
                    elif b == node["id"]:
                        connected_nodes.append(a)
                
                report_data.append({
                    "node_id": node["id"],
                    "type": node["type"],
                    "x": node["x"],
                    "y": node["y"],
                    "x_m": node["x_m"],
                    "y_m": node["y_m"],
                    "region": region,
                    "degree": node["degree"],
                    "connected_nodes": ", ".join(connected_nodes),
                    "is_door": False
                })
        
        # Puertas
        for door in self.graph["doors"]:
            connected_nodes = []
            for a, b in self.graph["connections"]["door"]:
                if a == door["id"]:
                    connected_nodes.append(b)
                elif b == door["id"]:
                    connected_nodes.append(a)
            
            report_data.append({
                "node_id": door["id"],
                "type": "door",
                "x": door["x"],
                "y": door["y"],
                "x_m": door["x_m"],
                "y_m": door["y_m"],
                "region": f"Connects {door['connects']}",
                "degree": len(connected_nodes),
                "connected_nodes": ", ".join(connected_nodes),
                "is_door": True
            })
        
        df_report = pd.DataFrame(report_data)
        df_report.sort_values(by=["is_door", "region", "type"], inplace=True)
        
        if output_file:
            if output_file.endswith('.csv'):
                df_report.to_csv(output_file, index=False)
            else:
                df_report.to_csv(output_file + '.csv', index=False)
        
        return df_report

    def generate_region_summary(self, output_file=None):
        """Genera un resumen estadístico por región y guarda en CSV"""
        
        # Parte 1: Resumen por regiones
        region_data = []
        
        for region_id, nodes in self.graph["regions"].items():
            # Calcular métricas de la región
            num_nodes = len(nodes)
            degrees = [node["degree"] for node in nodes]
            types = [node["type"] for node in nodes]
            
            # Encontrar puertas asociadas a esta región
            associated_doors = []
            for door in self.graph["doors"]:
                if region_id in door["connects"]:
                    associated_doors.append(door["id"])
            
            region_data.append({
                "region_id": region_id,
                "num_nodes": num_nodes,
                "avg_degree": np.mean(degrees),
                "num_endpoints": types.count("endpoint"),
                "num_junctions": types.count("junction"),
                "num_intermediate": types.count("intermediate"),
                "associated_doors": ", ".join(associated_doors),
                "centroid": self.centroides[region_id-2] if region_id-2 < len(self.centroides) else None
            })
        
        # Crear DataFrame de regiones
        region_df = pd.DataFrame(region_data)
        
        # Parte 2: Resumen general del grafo
        general_data = {
            "total_regiones": [len(self.graph["regions"])],
            "total_puertas": [len(self.graph["doors"])],
            "total_nodos": [sum(len(nodes) for nodes in self.graph["regions"].values())],
            "conexiones_internas": [len(self.graph["connections"]["internal"])],
            "conexiones_puertas": [len(self.graph["connections"]["door"])],
            "fecha_generacion": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        general_df = pd.DataFrame(general_data)
        
        # Guardar a CSV con timestamp para evitar sobrescribir
        if output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            regions_filename = f"{output_file}_regions_{timestamp}.csv"
            general_filename = f"{output_file}_general_{timestamp}.csv"
            
            region_df.to_csv(regions_filename, index=False)
            general_df.to_csv(general_filename, index=False)
            
            print(f"Datos de regiones guardados en: {regions_filename}")
            print(f"Datos generales guardados en: {general_filename}")
        
        # Imprimir resumen en consola
        print("\nEstructura del grafo:")
        print(f"Regiones: {general_data['total_regiones'][0]}")
        print(f"Puertas: {general_data['total_puertas'][0]}")
        print(f"Nodos: {general_data['total_nodos'][0]}")
        print(f"Conexiones internas: {general_data['conexiones_internas'][0]}")
        print(f"Conexiones a puertas: {general_data['conexiones_puertas'][0]}")
        
        return region_df
    
    def generate_simple_skeleton_report(self, output_file=None):
        """Genera reporte simplificado del esqueleto"""
        node_data = []
        for node in self.skeleton_graph.nodes():
            y, x = self.coord_subnodos[node]
            degree = self.skeleton_graph.degree[node]
            node_type = "endpoint" if degree == 1 else "intermediate" if degree == 2 else "junction"
            
            node_data.append({
                "node_id": node,
                "x": x,
                "y": y,
                "type": node_type,
                "degree": degree,
                "region": self.regionSubNodos[node] if self.regionSubNodos[node] != -1 else "border"
            })
        
        connection_data = []
        for u, v in self.skeleton_graph.edges():
            y1, x1 = self.coord_subnodos[u]
            y2, x2 = self.coord_subnodos[v]
            dist = distance.euclidean((x1, y1), (x2, y2))
            
            connection_data.append({
                "node1": u,
                "node2": v,
                "distance": f"{dist:.2f} px",
                "same_region": self.regionSubNodos[u] == self.regionSubNodos[v]
            })
        
        df_nodes = pd.DataFrame(node_data)
        df_connections = pd.DataFrame(connection_data)
        
        if output_file:
            if output_file.endswith('.csv'):
                df_nodes.to_csv(f"{output_file}_nodes.csv", index=False)
                df_connections.to_csv(f"{output_file}_connections.csv", index=False)
            else :
                df_nodes.to_csv(f"{output_file}_nodes.csv", index=False)
                df_connections.to_csv(f"{output_file}_connections.csv", index=False)
        
        return {
            "nodes": df_nodes,
            "connections": df_connections
        }

class LinearTopologyOptimizer:
    class Config:
        def __init__(self):
            # Node selection
            self.keep_doors = True
            self.keep_endpoints = True
            self.min_node_degree = 2
            self.keep_high_traffic = True
            
            # Edge control
            self.max_edges_per_node = 2
            self.connection_radius = 50.0
            self.min_path_length = 3
            self.max_path_deviation = 1.5
            
            # Geometry optimization
            self.merge_aligned_nodes = True
            self.alignment_tolerance = 15.0
            self.min_line_length = 30.0
            self.max_merge_distance = 20.0
            
            # Region handling
            self.enforce_region_connectivity = True
            self.min_region_connections = 1
    
    def __init__(self, original_graph, config=None):
        self.original_graph = original_graph
        self.config = config if config else self.Config()
        self._validate_input()
        
        # Precompute region mappings
        self.node_to_region = self._build_node_region_map()
        self.region_doors = self._identify_region_doors()
    
    def _validate_input(self):
        """Validación básica del grafo de entrada"""
        required_sections = {'regions', 'doors', 'connections', 'meta'}
        if not required_sections.issubset(self.original_graph.keys()):
            raise ValueError("El grafo original no tiene la estructura esperada")

    def _build_node_region_map(self):
        """Mapea cada nodo a su región"""
        node_region = {}
        for region_id, nodes in self.original_graph['regions'].items():
            for node in nodes:
                node_region[node['id']] = region_id
        for door in self.original_graph['doors']:
            node_region[door['id']] = 'door'
        return node_region

    def _build_base_graph(self):
        """Construye el grafo NetworkX inicial"""
        G = nx.Graph()
        
        # Añadir nodos de regiones
        for region_id, nodes in self.original_graph['regions'].items():
            for node in nodes:
                # Crear copia del nodo sin 'region' si ya existe
                node_attrs = node.copy()
                if 'region' in node_attrs:
                    del node_attrs['region']
                G.add_node(node['id'], **node_attrs, region=region_id)
        
        # Añadir nodos puerta
        for door in self.original_graph['doors']:
            # Crear copia del nodo sin 'region' si ya existe
            door_attrs = door.copy()
            if 'region' in door_attrs:
                del door_attrs['region']
            G.add_node(door['id'], **door_attrs, region='door')
        
        # Añadir conexiones internas
        for src, tgt in self.original_graph['connections'].get('internal', []):
            G.add_edge(src, tgt, type='internal')
        
        # Añadir conexiones puerta
        for src, tgt in self.original_graph['connections'].get('door', []):
            G.add_edge(src, tgt, type='door')
        
        return G

    def _identify_region_doors(self):
        """Identifica qué puertas conectan qué regiones"""
        region_pairs = defaultdict(set)
        G = self._build_base_graph()
        
        for door in self.original_graph['doors']:
            door_id = door['id']
            neighbors = list(G.neighbors(door_id))
            for neighbor in neighbors:
                region = self.node_to_region.get(neighbor)
                if region and region != 'door':
                    for other_neighbor in neighbors:
                        other_region = self.node_to_region.get(other_neighbor)
                        if other_region and other_region != region and other_region != 'door':
                            pair = tuple(sorted((region, other_region)))
                            region_pairs[pair].add(door_id)
        
        return dict(region_pairs)

    def optimize(self):
        """Proceso completo de optimización"""
        # 1. Construir grafo base
        base_graph = self._build_base_graph()
        
        # 2. Seleccionar nodos clave
        key_nodes = self._select_key_nodes(base_graph)
        
        # 3. Asegurar conectividad entre regiones
        if self.config.enforce_region_connectivity:
            self._ensure_region_connectivity(base_graph, key_nodes)
        
        # 4. Construir grafo lineal
        linear_graph = self._build_linear_graph(base_graph, key_nodes)
        
        # 5. Simplificación geométrica
        if self.config.merge_aligned_nodes:
            self._simplify_with_line_detection(linear_graph)
            
        # 6. Verificación final
        self._verify_connectivity(linear_graph) 

        return self._format_output(linear_graph)

    def _select_key_nodes(self, G):
        """Selecciona nodos importantes basados en la configuración"""
        key_nodes = set()
        
        # Conservar puertas si está configurado
        if self.config.keep_doors:
            key_nodes.update(n for n in G.nodes if n.startswith('D'))
        
        # Conservar nodos finales
        if self.config.keep_endpoints:
            key_nodes.update(n for n in G.nodes if G.degree(n) == 1)
        
        # Conservar nodos con suficiente grado (asegurando min_node_degree es int)
        min_degree = self.config.min_node_degree
        if isinstance(min_degree, tuple):  # Si accidentalmente es una tupla
            min_degree = min_degree[0]  # Tomamos el primer elemento
        key_nodes.update(n for n in G.nodes if G.degree(n) >= min_degree)
        
        # Añadir nodos de alto tráfico basado en centralidad
        if self.config.keep_high_traffic and len(G.nodes) > 2:
            betweenness = nx.betweenness_centrality(G)
            avg_betweenness = sum(betweenness.values()) / len(betweenness)
            key_nodes.update(n for n, score in betweenness.items() if score > 2 * avg_betweenness)
        
        return key_nodes

    def _ensure_region_connectivity(self, G, key_nodes):
        """Asegura conectividad mínima entre regiones"""
        for (r1, r2), doors in self.region_doors.items():
            connecting_doors = doors & key_nodes
            if not connecting_doors:
                # Añadir la puerta más central
                betweenness = nx.betweenness_centrality_subset(G, doors, doors)
                best_door = max(betweenness.items(), key=lambda x: x[1])[0]
                key_nodes.add(best_door)

    def _build_linear_graph(self, G, key_nodes):
        """Construye el grafo lineal con conexiones optimizadas"""
        linear = nx.Graph()
        node_positions = {n: (G.nodes[n]['x'], G.nodes[n]['y']) for n in G.nodes}
        
        # Añadir nodos clave
        for node in key_nodes:
            linear.add_node(node, **G.nodes[node])
        
        # Construir grafo de proximidad
        proximity_graph = self._build_proximity_graph(G, key_nodes, node_positions)
        
        # Construir árbol de expansión mínima
        self._build_mst(linear, G, proximity_graph)
        
        # Añadir conexiones adicionales de calidad
        self._add_quality_connections(linear, G, proximity_graph)
        
        return linear

    def _build_proximity_graph(self, G, key_nodes, node_positions):
        """Grafo completo con posibles conexiones válidas"""
        prox_graph = nx.Graph()
        prox_graph.add_nodes_from(key_nodes)
        
        # Aseguramos que los parámetros sean numéricos
        min_path_len = int(self.config.min_path_length)
        connection_radius = float(self.config.connection_radius)
        max_deviation = float(self.config.max_path_deviation)
        
        for u, v in combinations(key_nodes, 2):
            path = self._find_optimal_path(G, u, v)
            if path:
                straight_dist = hypot(
                    node_positions[u][0] - node_positions[v][0],
                    node_positions[u][1] - node_positions[v][1]
                )
                path_length = sum(
                    hypot(
                        node_positions[path[i]][0] - node_positions[path[i+1]][0],
                        node_positions[path[i]][1] - node_positions[path[i+1]][1]
                    )
                    for i in range(len(path)-1)
                )
                
                straightness = straight_dist / path_length if path_length > 0 else 0
                
                if (straightness >= 0.9 and
                    len(path) >= min_path_len and
                    path_length <= connection_radius * max_deviation):
                    
                    prox_graph.add_edge(u, v, 
                                      weight=len(path),
                                      path=path,
                                      straightness=straightness,
                                      length=path_length)
        
        return prox_graph

    def _build_mst(self, linear, G, proximity_graph):
        """Construye árbol de expansión mínima"""
        if not proximity_graph.edges():
            return
            
        edges_with_metrics = [
            (u, v, {
                'weight': data['weight'] * (1 - data['straightness']),
                'original_weight': data['weight'],
                'path': data['path']
            })
            for u, v, data in proximity_graph.edges(data=True)
        ]
        
        mst_graph = nx.Graph()
        mst_graph.add_weighted_edges_from([
            (u, v, metric['weight']) for u, v, metric in edges_with_metrics
        ])
        
        mst = nx.minimum_spanning_tree(mst_graph)
        
        for u, v in mst.edges():
            edge_data = next(
                data for (nu, nv, data) in edges_with_metrics 
                if (nu == u and nv == v) or (nu == v and nv == u)
            )
            linear.add_edge(u, v, 
                          weight=edge_data['original_weight'],
                          path=edge_data['path'])

    def _add_quality_connections(self, linear, G, proximity_graph):
        """Añade conexiones adicionales de alta calidad"""
        edges = sorted(
            proximity_graph.edges(data=True),
            key=lambda x: (-x[2]['straightness'], x[2]['length'])
        )
        
        for u, v, data in edges:
            if (linear.degree(u) < self.config.max_edges_per_node and 
                linear.degree(v) < self.config.max_edges_per_node and 
                not linear.has_edge(u, v)):
                
                linear.add_edge(u, v, 
                              weight=data['weight'],
                              path=data['path'])

    def _find_optimal_path(self, G, start, end):
        """Encuentra el mejor camino considerando restricciones de región"""
        start_region = self.node_to_region.get(start)
        end_region = self.node_to_region.get(end)
        
        if start_region != end_region and start_region != 'door' and end_region != 'door':
            # Debe pasar por una puerta conectiva
            region_pair = tuple(sorted((start_region, end_region)))
            connecting_doors = self.region_doors.get(region_pair, set())
            
            if not connecting_doors:
                return None
                
            shortest_path = None
            for door in connecting_doors:
                try:
                    path1 = nx.shortest_path(G, start, door)
                    path2 = nx.shortest_path(G, door, end)
                    full_path = path1[:-1] + path2
                    if shortest_path is None or len(full_path) < len(shortest_path):
                        shortest_path = full_path
                except nx.NetworkXNoPath:
                    continue
                    
            return shortest_path
        else:
            try:
                return nx.shortest_path(G, start, end)
            except nx.NetworkXNoPath:
                return None

    def _simplify_with_line_detection(self, G):
        """Simplificación mediante detección de líneas rectas"""
        self._merge_close_nodes(G)
        lines = self._detect_straight_lines(G)
        self._simplify_detected_lines(G, lines)

    def _merge_close_nodes(self, G):
        """Fusiona nodos muy cercanos"""
        nodes = list(G.nodes)
        positions = {n: (G.nodes[n]['x'], G.nodes[n]['y']) for n in nodes}
        to_merge = []
        
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                dist = hypot(
                    positions[u][0] - positions[v][0],
                    positions[u][1] - positions[v][1]
                )
                if dist < self.config.max_merge_distance:
                    if u.startswith('D') or (not v.startswith('D') and G.degree(u) >= G.degree(v)):
                        to_merge.append((v, u))
                    else:
                        to_merge.append((u, v))
        
        for source, target in to_merge:
            if source in G and target in G:
                for neighbor in list(G.neighbors(source)):
                    if neighbor != target and not G.has_edge(target, neighbor):
                        G.add_edge(target, neighbor)
                G.remove_node(source)

    def _detect_straight_lines(self, G):
        """Detecta secuencias de nodos alineados"""
        lines = []
        visited = set()
        positions = {n: (G.nodes[n]['x'], G.nodes[n]['y']) for n in G.nodes}
        
        for node in G.nodes:
            if node in visited or G.degree(node) != 2:
                continue
                
            line = [node]
            visited.add(node)
            
            # Explorar en una dirección
            current = node
            while True:
                neighbors = [n for n in G.neighbors(current) if n not in visited]
                if len(neighbors) != 1:
                    break
                    
                next_node = neighbors[0]
                if G.degree(next_node) != 2:
                    break
                    
                if len(line) >= 1:
                    angle = self._calculate_angle(
                        positions[line[-1]], 
                        positions[current], 
                        positions[next_node]
                    )
                    if abs(angle - 180) > self.config.alignment_tolerance:
                        break
                
                line.append(next_node)
                visited.add(next_node)
                current = next_node
            
            # Explorar en la otra dirección
            current = node
            while True:
                neighbors = [n for n in G.neighbors(current) if n not in visited]
                if len(neighbors) != 1:
                    break
                    
                prev_node = neighbors[0]
                if G.degree(prev_node) != 2:
                    break
                    
                if len(line) >= 1:
                    angle = self._calculate_angle(
                        positions[prev_node], 
                        positions[current], 
                        positions[line[0]]
                    )
                    if abs(angle - 180) > self.config.alignment_tolerance:
                        break
                
                line.insert(0, prev_node)
                visited.add(prev_node)
                current = prev_node
            
            if len(line) >= 3:
                total_length = sum(
                    hypot(
                        positions[line[i]][0] - positions[line[i+1]][0],
                        positions[line[i]][1] - positions[line[i+1]][1]
                    )
                    for i in range(len(line)-1)
                )
                if total_length >= self.config.min_line_length:
                    lines.append(line)
        
        return lines

    def _simplify_detected_lines(self, G, lines):
        """Simplifica las líneas detectadas"""
        for line in lines:
            if len(line) < 3:
                continue
                
            to_keep = {line[0], line[-1]}
            
            if len(line) > 4:
                mid_index = len(line) // 2
                to_keep.add(line[mid_index])
            
            for node in line:
                if node not in to_keep and node in G:
                    neighbors = list(G.neighbors(node))
                    for neighbor in neighbors:
                        if neighbor not in line:
                            closest = min(
                                to_keep, 
                                key=lambda x: hypot(
                                    G.nodes[x]['x'] - G.nodes[neighbor]['x'],
                                    G.nodes[x]['y'] - G.nodes[neighbor]['y']
                                )
                            )
                            if not G.has_edge(closest, neighbor):
                                G.add_edge(closest, neighbor)
                    
                    G.remove_node(node)
            
            kept_ordered = [n for n in line if n in to_keep]
            for i in range(len(kept_ordered)-1):
                if not G.has_edge(kept_ordered[i], kept_ordered[i+1]):
                    G.add_edge(kept_ordered[i], kept_ordered[i+1])

    def _calculate_angle(self, a, b, c):
        """Calcula el ángulo ABC en grados"""
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        
        angle_ba = atan2(ba[1], ba[0])
        angle_bc = atan2(bc[1], bc[0])
        angle = degrees(angle_bc - angle_ba)
        
        return angle if angle >= 0 else angle + 360

    def _verify_connectivity(self, G):
        """Verifica que se mantenga la conectividad requerida"""
        if self.config.enforce_region_connectivity:
            for (r1, r2), doors in self.region_doors.items():
                region1_nodes = [n for n in G.nodes if self.node_to_region.get(n) == r1]
                region2_nodes = [n for n in G.nodes if self.node_to_region.get(n) == r2]
                
                if region1_nodes and region2_nodes:
                    connected = False
                    for door in doors:
                        if door in G.nodes:
                            has_r1 = any(self.node_to_region.get(n) == r1 for n in G.neighbors(door))
                            has_r2 = any(self.node_to_region.get(n) == r2 for n in G.neighbors(door))
                            if has_r1 and has_r2:
                                connected = True
                                break
                    
                    if not connected and doors:
                        best_door = max(
                            (door for door in doors if door in G.nodes),
                            key=lambda x: nx.betweenness_centrality(G).get(x, 0),
                            default=None
                        )
                        
                        if best_door:
                            closest_r1 = min(
                                region1_nodes,
                                key=lambda x: hypot(
                                    G.nodes[x]['x'] - G.nodes[best_door]['x'],
                                    G.nodes[x]['y'] - G.nodes[best_door]['y']
                                )
                            )
                            closest_r2 = min(
                                region2_nodes,
                                key=lambda x: hypot(
                                    G.nodes[x]['x'] - G.nodes[best_door]['x'],
                                    G.nodes[x]['y'] - G.nodes[best_door]['y']
                                )
                            )
                            
                            if not G.has_edge(best_door, closest_r1):
                                G.add_edge(best_door, closest_r1)
                            if not G.has_edge(best_door, closest_r2):
                                G.add_edge(best_door, closest_r2)

    def _format_output(self, G):
        """Formatea el grafo optimizado para salida"""
        nodes_output = {}
        for node_id, attrs in G.nodes(data=True):
            node_data = attrs.copy()
            node_data['region'] = self.node_to_region.get(node_id, 'unknown')
            nodes_output[node_id] = node_data
        
        edges_output = []
        for u, v, data in G.edges(data=True):
            edge_data = {
                'source': u,
                'target': v,
                'weight': data.get('weight', 1),
                'path': data.get('path', [u, v])
            }
            edges_output.append(edge_data)
        
        return {
            "meta": {
                **self.original_graph['meta'],
                "optimizer": "linear_topology",
                "nodes": len(G.nodes),
                "edges": len(G.edges),
                "parameters": {k: v for k, v in vars(self.config).items() if not k.startswith('_')}
            },
            "nodes": nodes_output,
            "edges": edges_output
        }


class GraphIO:
    def __init__(self):
        pass
    
    def save_graph(self, optimized_graph: dict, file_path: str, compressed: bool = True):
        """Guarda el grafo optimizado en un archivo"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        graph_to_save = self._convert_numpy_types(optimized_graph)
        
        if compressed:
            if not path.suffix == '.gz':
                path = path.with_suffix(path.suffix + '.gz')
            with gzip.open(path, 'wt', encoding='utf-8') as f:
                json.dump(graph_to_save, f, indent=2, default=self._json_serializer)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(graph_to_save, f, indent=2, default=self._json_serializer)
    
    def load_graph(self, file_path: str) -> dict:
        """Carga un grafo optimizado desde archivo"""
        path = Path(file_path)
        
        if path.suffix == '.gz' or path.suffixes[-1] == '.gz':
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    def save_graph_pickle(self, optimized_graph: dict, file_path: str):
        """Guarda el grafo optimizado usando pickle"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(optimized_graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_graph_pickle(self, file_path: str) -> dict:
        """Carga un grafo optimizado usando pickle"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def _convert_numpy_types(self, obj):
        """Convierte tipos NumPy a tipos nativos de Python"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        return obj
    
    def _json_serializer(self, obj):
        """Serializador personalizado para JSON"""
        if isinstance(obj, (np.integer, np.floating)):
            return self._convert_numpy_types(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    def visualize_graph(self,optimized_graph, show_all_labels=True, label_threshold=20, path=None):   
        """
        Visualización mejorada con control de etiquetas
        
        Args:
            optimized_graph: Grafo optimizado
            show_all_labels: Mostrar todos los labels si es True
            label_threshold: Máximo de labels a mostrar si show_all_labels es False
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        from matplotlib import patheffects  # Importación añadida

        # Configuración de estilos
        plt.style.use('seaborn-whitegrid')
        
        # Crear grafo
        G = nx.Graph()
        
        # Añadir nodos y conexiones
        for node_id, attrs in optimized_graph["nodes"].items():
            G.add_node(node_id, **attrs)
        for edge in optimized_graph["edges"]:
            G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1))
        
        # Posiciones y tipos de nodos
        pos = {node: (attrs['x'], attrs['y']) for node, attrs in optimized_graph["nodes"].items()}

        # Convertir posiciones a metros para las etiquetas
        pos_metros = {node: (attrs['x_m'], attrs['y_m']) for node, attrs in optimized_graph["nodes"].items()}

        door_nodes = [n for n in G.nodes if n.startswith('D')]
        region_nodes = [n for n in G.nodes if not n.startswith('D')]
        
        # Configuración de figura
        fig, ax = plt.subplots(figsize=(12, 10))

        # Dibujar conexiones
        nx.draw_networkx_edges(
            G, pos,
            width=1.2,
            edge_color='#607c8e',
            alpha=0.6,
            ax=ax
        )
        
        # Dibujar nodos con estilo mejorado
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=door_nodes,
            node_color='#ff6b6b',
            node_size=200,
            edgecolors='#c92a2a',
            linewidths=1.5,
            label='Puertas',
            ax=ax
        )
        
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=region_nodes,
            node_color='#4dabf7',
            node_size=80,
            edgecolors='#1971c2',
            linewidths=0.8,
            label='Nodos Región',
            ax=ax
        )
        
        # Selección de nodos a etiquetar
        if show_all_labels or len(G.nodes) <= label_threshold:
            # labels = {node: node for node in G.nodes}
            # Crear etiquetas con nombre de nodo y coordenadas en metros
            labels = {
                node: f"{node}\n({pos_metros[node][0]:.2f}, {pos_metros[node][1]:.2f})"
                for node in G.nodes
            }
        else:
            # Seleccionar nodos importantes para etiquetar
            important_nodes = set(door_nodes)
            
            # Añadir nodos con alta centralidad
            centrality = nx.degree_centrality(G)
            top_central = sorted(centrality.items(), key=lambda x: -x[1])[:label_threshold//2]
            important_nodes.update([n for n, _ in top_central])
            
            # Añadir nodos aleatorios para completar
            remaining_nodes = set(G.nodes) - important_nodes
            if remaining_nodes:
                import random
                important_nodes.update(random.sample(remaining_nodes, min(len(remaining_nodes), label_threshold-len(important_nodes))))
            
            # labels = {node: node for node in important_nodes}
            labels = {
                node: f"{node}\n({pos_metros[node][0]:.2f}, {pos_metros[node][1]:.2f})"
                for node in important_nodes
            }

        
        # Dibujar etiquetas con sombra para mejor legibilidad
        text = nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=8,
            font_family='monospace',
            font_weight='medium',
            font_color='#2b2d42',
            bbox=dict(
                alpha=0.7,
                facecolor='white',
                edgecolor='none',
                boxstyle='round,pad=0.2'
            ),
            ax=ax
        )
        
        # Mejorar legibilidad de etiquetas
        for _, t in text.items():
            t.set_rotation(20)
            t.set_path_effects([
                patheffects.withStroke(
                    linewidth=3,
                    foreground='white'
                )
            ])
        
        # Leyenda y título
        plt.legend(scatterpoints=1, frameon=True, loc='best')
        plt.title(
            f"Grafo Topológico optimizado\nTotal nodos: {len(G.nodes)} | Conexiones: {len(G.edges)}",
            fontsize=12,
            pad=20
        )
        
        # Cuadrícula y ejes
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.axis('equal')
        
        # Ajustar márgenes
        x_values = [x for x, _ in pos.values()]
        y_values = [y for _, y in pos.values()]
        plt.xlim(min(x_values)-50, max(x_values)+50)
        plt.ylim(min(y_values)-50, max(y_values)+50)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        if path:
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()


class PathPlanner:
    def __init__(self, optimized_graph: Dict):
        """
        Inicializa el planificador con el grafo optimizado
        
        Args:
            optimized_graph: El grafo optimizado en formato de diccionario
        """
        self.graph = optimized_graph
        self.node_data = optimized_graph['nodes']
        self.edge_data = optimized_graph['edges']
        
        # Preprocesamiento para crear estructuras eficientes
        self._build_adjacency_structure()
    
    def _build_adjacency_structure(self):
        """Construye una estructura de adyacencia eficiente para búsqueda"""
        self.adjacency = defaultdict(list)
        self.edge_info = {}
        
        for edge in self.edge_data:
            src = edge['source']
            tgt = edge['target']
            weight = edge.get('weight', 1)
            path = edge.get('path', [src, tgt])
            
            # Grafo no dirigido - añadimos conexión en ambos sentidos
            self.adjacency[src].append(tgt)
            self.adjacency[tgt].append(src)
            
            # Almacenamos información de la arista en ambos sentidos
            self.edge_info[(src, tgt)] = {
                'weight': weight,
                'path': path
            }
            self.edge_info[(tgt, src)] = {
                'weight': weight,
                'path': list(reversed(path))
            }
    
    def heuristic(self, node_a: str, node_b: str) -> float:
        """Función heurística para A* (distancia euclidiana entre nodos)"""
        pos_a = (self.node_data[node_a]['x'], self.node_data[node_a]['y'])
        pos_b = (self.node_data[node_b]['x'], self.node_data[node_b]['y'])
        return sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)
    
    def get_edge_weight(self, u: str, v: str) -> float:
        """Obtiene el peso de la arista entre dos nodos"""
        return self.edge_info.get((u, v), {}).get('weight', 1.0)
    
    def get_node_weight(self, node: str) -> float:
        """Obtiene el peso de un nodo (si está definido)"""
        return self.node_data.get(node, {}).get('weight', 1.0)
    
    def a_star_search(self, start: str, goal: str) -> Tuple[List[str], float]:
        """Implementación del algoritmo A* para encontrar el camino óptimo"""
        if start not in self.node_data or goal not in self.node_data:
            raise ValueError("Nodo inicial o final no existe en el grafo")
        
        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {node: float('inf') for node in self.node_data}
        g_score[start] = 0
        f_score = {node: float('inf') for node in self.node_data}
        f_score[start] = self.heuristic(start, goal)
        
        open_set_hash = {start}
        
        while open_set:
            current = heappop(open_set)[1]
            open_set_hash.remove(current)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                
                total_cost = 0
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    total_cost += self.get_edge_weight(u, v)
                    total_cost += self.get_node_weight(v)
                
                return path, total_cost
            
            for neighbor in self.adjacency[current]:
                tentative_g_score = g_score[current] + self.get_edge_weight(current, neighbor) + self.get_node_weight(neighbor)
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    if neighbor not in open_set_hash:
                        heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        raise ValueError("No se encontró camino entre los nodos especificados")
    
    def get_detailed_path(self, start: str, goal: str) -> Dict:
        """Obtiene un camino detallado con todos los puntos intermedios"""
        key_nodes_path, total_cost = self.a_star_search(start, goal)
        
        detailed_path = []
        subpaths = []
        
        for i in range(len(key_nodes_path)-1):
            u = key_nodes_path[i]
            v = key_nodes_path[i+1]
            
            subpath = self.edge_info[(u, v)]['path']
            
            if detailed_path:
                subpath = subpath[1:]
            
            detailed_path.extend(subpath)
            subpaths.append(subpath)
        
        return {
            'path': detailed_path,
            'subpaths': subpaths,
            'waypoints': key_nodes_path,
            'total_cost': total_cost,
            'length': len(detailed_path)-1,
            'start': start,
            'goal': goal
        }
    
    def visualize_path(self, detailed_path: Dict, show_full_graph: bool = True, figsize: tuple = (12, 10), path=None):
        """Visualiza el camino encontrado en el grafo"""
        plt.figure(figsize=figsize)
        ax = plt.gca()
        
        pos = {node: (data['x'], data['y']) for node, data in self.node_data.items()}
        pos_metros = {node: (data['x_m'], data['y_m']) for node, data in self.node_data.items()}
        G = nx.Graph()
        
        for node, attrs in self.node_data.items():
            G.add_node(node, **attrs)
        
        for edge in self.edge_data:
            G.add_edge(edge['source'], edge['target'], weight=edge.get('weight', 1))
        
        if show_full_graph:
            nodes_to_show = G.nodes()
            edges_to_show = G.edges()
        else:
            nodes_to_show = set(detailed_path['path'])
            for node in detailed_path['path']:
                nodes_to_show.update(G.neighbors(node))
            edges_to_show = [
                (u, v) for u, v in G.edges() 
                if u in nodes_to_show and v in nodes_to_show
            ]
        
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[n for n in nodes_to_show if n not in detailed_path['path']],
            node_size=50,
            node_color='lightgray',
            edgecolors='gray',
            linewidths=0.5,
            ax=ax
        )
        
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[e for e in edges_to_show if e not in zip(detailed_path['path'][:-1], detailed_path['path'][1:])],
            width=0.5,
            edge_color='lightgray',
            alpha=0.6,
            ax=ax
        )
        
        path_edges = list(zip(detailed_path['path'][:-1], detailed_path['path'][1:]))
        cmap = plt.cm.plasma
        path_colors = [cmap(i) for i in np.linspace(0, 1, len(detailed_path['path']))]
        
        for i, node in enumerate(detailed_path['path']):
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[node],
                node_size=100 if node in detailed_path['waypoints'] else 70,
                node_color=[path_colors[i]],
                edgecolors='darkred' if node in detailed_path['waypoints'] else 'red',
                linewidths=1.5,
                ax=ax
            )
        
        for i, (u, v) in enumerate(path_edges):
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=2,
                edge_color=cmap(i/len(path_edges)),
                alpha=0.8,
                ax=ax
            )
        
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[detailed_path['start']],
            node_size=300,
            node_color='lime',
            edgecolors='darkgreen',
            linewidths=2,
            ax=ax
        )
        
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[detailed_path['goal']],
            node_size=300,
            node_color='red',
            edgecolors='darkred',
            linewidths=2,
            ax=ax
        )
        
        labels = {
            detailed_path['start']: 'START: ' + detailed_path['start'],
            detailed_path['goal']: 'GOAL: ' + detailed_path['goal']
        }
        
        for node in detailed_path['waypoints'][1:-1]:
            # labels[node] = node
            labels[node] = f"{node}\n({pos_metros[node][0]:.2f}, {pos_metros[node][1]:.2f})"
        
        text = nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=10,
            font_family='sans-serif',
            font_weight='bold',
            ax=ax
        )
        
        for _, t in text.items():
            t.set_path_effects([
                patheffects.withStroke(linewidth=3, foreground='white')
            ])
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(detailed_path['path'])-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Progreso del camino', rotation=270, labelpad=15)
        
        title = f"Planificación de Trayectoria\n{detailed_path['start']} → {detailed_path['goal']}"
        subtitle = f"Costo total: {detailed_path['total_cost']:.2f} | Segmentos: {detailed_path['length']}"
        plt.title(f"{title}\n{subtitle}", fontsize=12, pad=20)
        
        ax.set_facecolor('#f5f5f5')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.axis('equal')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        if path:
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()