"""Kiểm thử số học, hình học, metric và tương thích pipeline."""

import cv2
import numpy as np
import pytest

from src.edge_detection import (
    gradient_magnitude_direction, laplacian_edge_detection, laplacian_response,
    normalize_edges, prewitt_edge_detection, prewitt_gradients,
    sobel_edge_detection, sobel_gradients,
)
from src.edge_metrics import evaluate_edges
from src.pipeline import pencil_sketch_pipeline
from src.smoothing import gaussian_filter
from src.thresholding import threshold_edges

DETECTORS = [sobel_edge_detection, prewitt_edge_detection, laplacian_edge_detection]


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize("value", [0, 127, 255])
def test_constant_has_no_false_edges(detector, value):
    result = detector(np.full((11, 13), value, dtype=np.uint8))
    assert result.shape == (11, 13)
    assert result.dtype == np.uint8
    assert not np.any(result)


@pytest.mark.parametrize("gradients,expected", [(sobel_gradients, 1020), (prewitt_gradients, 765)])
@pytest.mark.parametrize("axis", [0, 1])
def test_step_has_exact_location_sign_and_strength(gradients, expected, axis):
    image = np.zeros((10, 10), dtype=np.uint8)
    if axis == 1:
        image[:, 5:] = 255
    else:
        image[5:, :] = 255
    gx, gy = gradients(image)
    active, inactive = (gx, gy) if axis == 1 else (gy, gx)
    truth = np.zeros((10, 10), dtype=np.float32)
    if axis == 1:
        truth[:, 4:6] = -expected
    else:
        truth[4:6, :] = -expected
    np.testing.assert_array_equal(active, truth)
    np.testing.assert_array_equal(inactive, 0)
    assert gx.dtype == gy.dtype == np.float32
    magnitude, direction = gradient_magnitude_direction(gx, gy)
    np.testing.assert_array_equal(magnitude, np.abs(truth))
    # So sánh vector đơn vị để không nhầm -pi và +pi ở cùng một hướng.
    mask = magnitude > 0
    np.testing.assert_allclose(np.cos(direction[mask]), -1 if axis == 1 else 0, atol=1e-6)
    np.testing.assert_allclose(np.sin(direction[mask]), 0 if axis == 1 else -1, atol=1e-6)
    np.testing.assert_array_equal(direction[~mask], 0)


def test_laplacian_impulse_by_hand():
    image = np.zeros((7, 7), dtype=np.uint8)
    image[3, 3] = 100
    expected = np.zeros((7, 7), dtype=np.float32)
    expected[3, 3] = -400
    expected[2, 3] = expected[4, 3] = expected[3, 2] = expected[3, 4] = 100
    np.testing.assert_array_equal(laplacian_response(image), expected)


@pytest.mark.parametrize("detector", DETECTORS)
def test_rectangle_outline_and_quiet_regions(detector):
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:24, 8:24] = 255
    edges = detector(image)
    assert np.all(edges[7:9, 9:23] > 0)
    assert np.all(edges[23:25, 9:23] > 0)
    assert np.all(edges[9:23, 7:9] > 0)
    assert np.all(edges[9:23, 23:25] > 0)
    assert not np.any(edges[10:22, 10:22])
    assert not np.any(edges[:6])


def test_direction_for_oblique_gradient():
    gx = np.array([[3, -3, 0]], dtype=np.float32)
    gy = np.array([[4, 4, 0]], dtype=np.float32)
    magnitude, direction = gradient_magnitude_direction(gx, gy)
    np.testing.assert_array_equal(magnitude, [[5, 5, 0]])
    np.testing.assert_allclose(direction, [[np.arctan2(4, 3), np.arctan2(4, -3), 0]])


def test_signed_gradients_match_opencv_including_border():
    image = np.random.default_rng(42).integers(0, 256, (19, 23), dtype=np.uint8)
    border = cv2.BORDER_REFLECT_101
    gx, gy = sobel_gradients(image)
    np.testing.assert_array_equal(gx, -cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3, borderType=border))
    np.testing.assert_array_equal(gy, -cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3, borderType=border))
    kernel = np.array([[-1, 0, 1]] * 3, dtype=np.float32)
    px, py = prewitt_gradients(image)
    np.testing.assert_array_equal(px, -cv2.filter2D(image, cv2.CV_32F, kernel, borderType=border))
    np.testing.assert_array_equal(py, -cv2.filter2D(image, cv2.CV_32F, kernel.T, borderType=border))
    np.testing.assert_array_equal(laplacian_response(image), cv2.Laplacian(image, cv2.CV_32F, ksize=1, borderType=border))


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (2, 2)])
def test_tiny_images(detector, shape):
    image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    result = detector(image)
    assert result.shape == shape
    assert result.dtype == np.uint8


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize("image,error", [
    ([[0, 1]], TypeError),
    (np.zeros((0, 3), dtype=np.uint8), ValueError),
    (np.zeros((3, 3, 3), dtype=np.uint8), ValueError),
    (np.zeros((3, 3), dtype=np.float32), ValueError),
])
def test_reject_invalid_gray(detector, image, error):
    with pytest.raises(error):
        detector(image)


def test_threshold_boundary_and_zero_image():
    image = np.array([[0, 59, 60, 61, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(threshold_edges(image, 60), [[0, 0, 0, 255, 255]])
    assert threshold_edges(image).dtype == np.uint8
    assert not np.any(threshold_edges(np.zeros((3, 3), dtype=np.uint8), 0))
    assert not np.any(threshold_edges(image, 255))


@pytest.mark.parametrize("threshold", [-1, 256, np.nan, np.inf])
def test_threshold_rejects_invalid_range(threshold):
    with pytest.raises(ValueError):
        threshold_edges(np.zeros((2, 2), dtype=np.uint8), threshold)


@pytest.mark.parametrize("threshold", [True, "60", None])
def test_threshold_rejects_invalid_type(threshold):
    with pytest.raises(TypeError):
        threshold_edges(np.zeros((2, 2), dtype=np.uint8), threshold)


def test_metrics_with_hand_counted_confusion_matrix():
    predicted = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool)
    reference = np.array([[1, 0, 1], [0, 1, 1]], dtype=bool)
    scores = evaluate_edges(predicted, reference)
    assert [scores[k] for k in ("tp", "fp", "fn", "tn")] == [2, 1, 2, 1]
    assert scores['precision'] == pytest.approx(2 / 3)
    assert scores['recall'] == pytest.approx(1 / 2)
    assert scores['f1'] == pytest.approx(4 / 7)
    assert scores['iou'] == pytest.approx(2 / 5)
    assert scores == evaluate_edges(predicted.astype(np.uint8) * 255, reference.astype(np.uint8) * 255)


@pytest.mark.parametrize("predicted,reference,expected", [(False, False, 1), (False, True, 0), (True, False, 0), (True, True, 1)])
def test_metrics_zero_denominators(predicted, reference, expected):
    scores = evaluate_edges(np.full((3, 4), predicted), np.full((3, 4), reference))
    assert all(scores[k] == expected for k in ("precision", "recall", "f1", "iou"))


def test_metrics_reject_gray_or_mismatched_shape():
    with pytest.raises(ValueError):
        evaluate_edges(np.array([[128]], dtype=np.uint8), np.array([[255]], dtype=np.uint8))
    with pytest.raises(ValueError):
        evaluate_edges(np.zeros((3, 3), dtype=bool), np.zeros((4, 3), dtype=bool))


def test_gaussian_reduces_noise_response_before_normalization():
    image = np.clip(128 + np.random.default_rng(5).normal(0, 20, (64, 64)), 0, 255).astype(np.uint8)
    raw, _ = gradient_magnitude_direction(*sobel_gradients(image))
    filtered, _ = gradient_magnitude_direction(*sobel_gradients(gaussian_filter(image, 5, 1.0)))
    assert filtered.mean() < raw.mean() * 0.5


def test_existing_sobel_pipeline_result_is_unchanged():
    image = np.random.default_rng(7).integers(0, 256, (21, 25, 3), dtype=np.uint8)
    from src.grayscale import rgb_to_grayscale
    from src.pencil_sketch import create_pencil_sketch
    gray = rgb_to_grayscale(image)
    gx, gy = sobel_gradients(gaussian_filter(gray))
    # Công thức Sobel nguyên bản, không gọi wrapper vừa sửa.
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    original_edges = np.clip(np.rint(magnitude / magnitude.max() * 255), 0, 255).astype(np.uint8)
    expected = create_pencil_sketch(gray, original_edges)
    actual = pencil_sketch_pipeline(image)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8


def test_image_roundtrip_with_vietnamese_path(tmp_path):
    from src.image_io import load_image, save_image
    folder = tmp_path / "ảnh_kiểm_thử"
    folder.mkdir()
    path = folder / "kết_quả.png"
    gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
    save_image(gray, path)
    rgb = load_image(path)
    np.testing.assert_array_equal(rgb, np.repeat(gray[:, :, None], 3, axis=2))


def test_empty_image_file_is_reported_cleanly(tmp_path):
    from src.image_io import load_image
    path = tmp_path / "empty.png"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="Không thể đọc ảnh"):
        load_image(path)
