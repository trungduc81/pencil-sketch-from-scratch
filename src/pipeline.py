from src.edge_detection import sobel_edge_detection
from src.grayscale import rgb_to_grayscale
from src.pencil_sketch import create_pencil_sketch
from src.smoothing import gaussian_filter


def pencil_sketch_pipeline(img_rgb,smoothing_kernel_size=5,smoothing_sigma=1.0,sketch_kernel_size=21,sketch_sigma=5.0):
    gray = rgb_to_grayscale(img_rgb)

    smoothed = gaussian_filter(gray,smoothing_kernel_size,smoothing_sigma)
    edges = sobel_edge_detection(smoothed)

    sketch = create_pencil_sketch(gray, edges,sketch_kernel_size,sketch_sigma)
    
    return sketch
