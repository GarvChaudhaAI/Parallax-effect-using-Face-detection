import pygame
import math
import numpy as np
import cv2
import mediapipe as mp
def detect_face_center_mediapipe(image):
    """
    Detect face center using MediaPipe
    Requires: pip install mediapipe
    """
    
    
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    face_centers = []
    
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)
        
        if results.detections:
            h, w = image.shape[:2]
            
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # Calculate center
                center_x = x + width // 2
                center_y = y + height // 2
                
                face_centers.append((center_x, center_y, x, y, width, height))
    
    return face_centers

pygame.init()
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
fov = 90
aspect_ratio = SCREEN_WIDTH / SCREEN_HEIGHT
near = 0.1
far = 1000.0

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
def get_pixel(x,y):
    return ((x+1)*SCREEN_WIDTH//2,(-y+1)*SCREEN_HEIGHT//2)
def draw_triangle(screen, triangle,color=(255, 255, 255)):
    pygame.draw.polygon(screen, color, [
        get_pixel(triangle.v1.x, triangle.v1.y),
        get_pixel(triangle.v2.x, triangle.v2.y),
        get_pixel(triangle.v3.x, triangle.v3.y)
    ])

class Vector:
    def __init__(self, x, y, z, w=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w # Homogeneous coordinate for 3D transformations
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z) 
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)
    def __truediv__(self, scalar):
        return Vector(self.x / scalar, self.y / scalar, self.z / scalar)
    def cross(self, other):
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    def length(self):
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5
    def normalize(self):
        length = self.length()
        if length == 0:
            return Vector(0, 0, 0)
        return Vector(self.x / length, self.y / length, self.z / length, self.w)
    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y}, {self.z})"
class Triangle:
    def __init__(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
    def normal(self):
        edge1 = self.v2 - self.v1
        edge2 = self.v3 - self.v1
        return edge1.cross(edge2).normalize()
    def __repr__(self):
        return f"Triangle({self.v1}, {self.v2}, {self.v3})"
class Mesh:
    def __init__(self, vertices, triangles):
        self.vertices = vertices
        self.triangles = triangles
        self.sort_triangles()
    def order(self, t):
        return (t.v1.x + t.v2.x + t.v3.x)**2 + (t.v1.y + t.v2.y + t.v3.y)**2 + (t.v1.z + t.v2.z + t.v3.z)**2
    def sort_triangles(self):
        self.triangles.sort(key=lambda t: self.order(t), reverse=False)
    def render(self, screen, projection_matrix, color=(255, 255, 255)):
        pass
    def draw_triangles(self, screen, projection_matrix, translation_matrix, color =(255, 255, 255)):
        for triangle in self.triangles:
            
            if triangle.normal().dot(translation_matrix.mul_vector(triangle.v1))<=0:
                continue
            
            shade = (triangle.normal().dot(light)+1)/2
            new_color = (color[0] * shade, color[1] * shade, color[2] * shade)

            projected_v1 = projection_matrix.mul_vector(translation_matrix.mul_vector(triangle.v1))
            projected_v2 = projection_matrix.mul_vector(translation_matrix.mul_vector(triangle.v2))
            projected_v3 = projection_matrix.mul_vector(translation_matrix.mul_vector(triangle.v3))
            projected_v1 = projected_v1/ projected_v1.w
            projected_v2 = projected_v2/ projected_v2.w
            projected_v3 = projected_v3/ projected_v3.w
            projected_triangle = Triangle(projected_v1, projected_v2, projected_v3)
            draw_triangle(screen, projected_triangle, new_color)
    def read_obj(self, filepath):
        with open(filepath, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    parts = line.split()
                    vertex = Vector(float(parts[1]),float(parts[2]), float(parts[3]))
                    self.vertices.append(vertex)
                elif line.startswith('f '):
                    parts = line.split()
                    v1 = self.vertices[int(parts[1]) - 1]
                    v2 = self.vertices[int(parts[2]) - 1]
                    v3 = self.vertices[int(parts[3]) - 1]
                    triangle = Triangle(v1, v2, v3)
                    self.triangles.append(triangle)
        self.sort_triangles()
    def __repr__(self):
        return f"Mesh({len(self.vertices)} vertices, {len(self.triangles)} triangles)"
class Matrix:
    def __init__(self, values):
        self.values = values
    def mul_vector(self, vector):
        x = self.values[0][0] * vector.x + self.values[1][0] * vector.y + self.values[2][0] * vector.z + self.values[3][0] * vector.w
        y = self.values[0][1] * vector.x + self.values[1][1] * vector.y + self.values[2][1] * vector.z + self.values[3][1] * vector.w
        z = self.values[0][2] * vector.x + self.values[1][2] * vector.y + self.values[2][2] * vector.z + self.values[3][2] * vector.w
        w = self.values[0][3] * vector.x + self.values[1][3] * vector.y + self.values[2][3] * vector.z + self.values[3][3] * vector.w
        return Vector(x, y, z , w)
class ProjectionMatrix(Matrix):
    def __init__(self, fov, aspect_ratio, near, far):
        f = 1 / ( math.tan(fov * 0.5 * 3.14159 / 180))
        self.values = np.array([
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, far/(far-near) , 1],
            [0, 0, -far*near/(far-near), 0]
        ])

class TranslationMatrix(Matrix):
    def __init__(self, tx, ty, tz):
        self.values = np.array([
            [ 1, 0, 0, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 1, 0],
            [tx,ty,tz, 1]
        ])
light = Vector(1, 2, 4).normalize()  # Direction of the light source
running = True
triangles = [
    # Bottom face (z = 0)
    [(0, 0, 0), (1, 0, 0), (1, 1, 0)],  # Triangle 1
    [(0, 0, 0), (1, 1, 0), (0, 1, 0)],  # Triangle 2
    
    # Top face (z = 1)
    [(0, 0, 1), (1, 1, 1), (1, 0, 1)],  # Triangle 3
    [(0, 0, 1), (0, 1, 1), (1, 1, 1)],  # Triangle 4
    
    # Front face (y = 0)
    [(0, 0, 0), (1, 0, 1), (1, 0, 0)],  # Triangle 5
    [(0, 0, 0), (0, 0, 1), (1, 0, 1)],  # Triangle 6
    
    # Back face (y = 1)
    [(0, 1, 0), (1, 1, 0), (1, 1, 1)],  # Triangle 7
    [(0, 1, 0), (1, 1, 1), (0, 1, 1)],  # Triangle 8
    
    # Left face (x = 0)
    [(0, 0, 0), (0, 1, 0), (0, 1, 1)],  # Triangle 9
    [(0, 0, 0), (0, 1, 1), (0, 0, 1)],  # Triangle 10
    
    # Right face (x = 1)
    [(1, 0, 0), (1, 1, 1), (1, 1, 0)],  # Triangle 11
    [(1, 0, 0), (1, 0, 1), (1, 1, 1)]   # Triangle 12
]
triangles = [Triangle(Vector(*v1), Vector(*v2), Vector(*v3)) for v1, v2, v3 in triangles]
mesh = Mesh(vertices=[],triangles=[])
mesh.read_obj('VideoShip.obj')  # Load the mesh from an OBJ file
count =0
cap = cv2.VideoCapture(0)
pos_x,pos_y=0,0
while running:
    ret, frame = cap.read()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0)) # Clear the screen with black
    haar_centers = detect_face_center_mediapipe(frame)
    try:
        move_x,move_y = haar_centers[0][0], haar_centers[0][1] if haar_centers else (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        move_x = 3*(move_x - frame.shape[0] // 2) / (frame.shape[0]// 2)
        move_y = 3*(move_y - frame.shape[1] // 2) / (frame.shape[1] // 2)
        pos_x,pos_y = 0.3*move_x+0.7*pos_x,0.3*move_y+0.7*pos_y
    except:
        pass
    projection_matrix = ProjectionMatrix(fov, aspect_ratio, near, far)
    translation_matrix = TranslationMatrix(-pos_x ,-pos_y, 5)  # Move the mesh back along the z-axis
    mesh.draw_triangles(screen, projection_matrix, translation_matrix, color=(255, 255, 255))
    pygame.display.flip()
    count+=1
    #print()  # Update the display
pygame.quit()