"""Tests for the 68-point facial landmark fitter.

No model file is loaded: ``_load_facemark`` is patched throughout, so the suite
stays offline and weight-free like the detector tests beside it.
"""

from pathlib import Path

import numpy as np
import pytest

from detector.detector import Detection
from face import landmarks as landmarks_module
from face.landmarks import FaceLandmarker

# iBUG-68 index ranges, repeated here so the tests pin the convention
# independently of the implementation's own constants.
_JAW = range(0, 17)
_RIGHT_EYEBROW = range(17, 22)
_LEFT_EYEBROW = range(22, 27)
_NOSE = range(27, 36)
_RIGHT_EYE = range(36, 42)
_LEFT_EYE = range(42, 48)
_MOUTH = range(48, 68)


class _FakeFacemark:
    """Stands in for ``cv2.face.Facemark``, recording the boxes it was given."""

    def __init__(self, landmarks: list[np.ndarray] | None, ok: bool = True) -> None:
        self._landmarks = landmarks
        self._ok = ok
        self.fitted_boxes: list[list[list[int]]] = []

    def fit(
        self, image: np.ndarray, faces: np.ndarray
    ) -> tuple[bool, list[np.ndarray] | None]:
        """Return the canned ``(ok, landmarks)`` pair."""
        self.fitted_boxes.append(np.asarray(faces).tolist())
        return self._ok, self._landmarks


def _indexed_face() -> np.ndarray:
    """Return one 1x68x2 face whose point ``i`` is ``(i, 100 + i)``.

    Encoding the index into the coordinates makes a mis-sliced group visible in
    the assertion itself rather than as an off-by-one nobody notices.

    Note these points are collinear, which is fine for grouping tests but
    degenerate for triangulation; mesh tests use :func:`_scattered_face`.
    """
    points = [(float(index), float(100 + index)) for index in range(68)]
    return np.array([points], dtype=np.float32)


def _scattered_face(seed: int = 0) -> np.ndarray:
    """Return one 1x68x2 face of non-collinear points in general position.

    Delaunay needs points that are not all on one line, so the grouping
    fixture's diagonal cannot be reused for mesh assertions.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 200, size=(1, 68, 2)).astype(np.float32)


@pytest.fixture
def image() -> np.ndarray:
    """Return a blank BGR frame."""
    return np.zeros((240, 320, 3), dtype=np.uint8)


@pytest.fixture
def box() -> Detection:
    """Return one face box in corner coordinates."""
    return Detection(x1=10, y1=20, x2=60, y2=90, confidence=0.9)


def _fitter(monkeypatch: pytest.MonkeyPatch, model: _FakeFacemark) -> FaceLandmarker:
    """Build a landmarker whose model loading is stubbed out."""
    monkeypatch.setattr(landmarks_module, "_load_facemark", lambda _path: model)
    return FaceLandmarker("models/face/lbfmodel.yaml")


def test_fit_groups_points_into_the_four_requested_features(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray, box: Detection
) -> None:
    """Eyes, eyebrows, nose and mouth each carry their own iBUG-68 slice."""
    model = _FakeFacemark([_indexed_face()])

    (result,) = _fitter(monkeypatch, model).fit(image, [box])

    assert result.right_eyebrow == [(index, 100 + index) for index in _RIGHT_EYEBROW]
    assert result.left_eyebrow == [(index, 100 + index) for index in _LEFT_EYEBROW]
    assert result.nose == [(index, 100 + index) for index in _NOSE]
    assert result.right_eye == [(index, 100 + index) for index in _RIGHT_EYE]
    assert result.left_eye == [(index, 100 + index) for index in _LEFT_EYE]
    assert result.mouth == [(index, 100 + index) for index in _MOUTH]


def test_fit_omits_the_jaw_contour_by_default(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray, box: Detection
) -> None:
    """Points 0-16 trace the jaw and stay out of the default response.

    The mesh mode reports them on request, so this is the regression test that
    opting in is still required and the default did not quietly widen.
    """
    model = _FakeFacemark([_indexed_face()])

    (result,) = _fitter(monkeypatch, model).fit(image, [box])

    assert result.jaw is None
    reported = {
        point
        for group in (
            result.right_eyebrow,
            result.left_eyebrow,
            result.nose,
            result.right_eye,
            result.left_eye,
            result.mouth,
        )
        for point in group
    }
    assert reported.isdisjoint({(index, 100 + index) for index in _JAW})


def test_right_groups_are_the_subjects_right_not_the_viewers(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray, box: Detection
) -> None:
    """In iBUG-68 the ``right`` features sit on the left of the image."""
    points = [(50.0, 50.0)] * 68
    for index in _RIGHT_EYEBROW:
        points[index] = (20.0, 30.0)
    for index in _LEFT_EYEBROW:
        points[index] = (60.0, 30.0)
    for index in _RIGHT_EYE:
        points[index] = (25.0, 40.0)
    for index in _LEFT_EYE:
        points[index] = (55.0, 40.0)
    model = _FakeFacemark([np.array([points], dtype=np.float32)])

    (result,) = _fitter(monkeypatch, model).fit(image, [box])

    assert max(x for x, _ in result.right_eyebrow) < min(
        x for x, _ in result.left_eyebrow
    )
    assert max(x for x, _ in result.right_eye) < min(x for x, _ in result.left_eye)


def test_fit_passes_boxes_as_x_y_width_height(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray, box: Detection
) -> None:
    """OpenCV wants ``x, y, w, h``; detections carry corner coordinates."""
    model = _FakeFacemark([_indexed_face()])

    _fitter(monkeypatch, model).fit(image, [box])

    assert model.fitted_boxes == [[[10, 20, 50, 70]]]


def test_fit_returns_one_result_per_face(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray, box: Detection
) -> None:
    """Two boxes in, two grouped landmark sets out."""
    model = _FakeFacemark([_indexed_face(), _indexed_face()])
    second = Detection(x1=100, y1=20, x2=150, y2=90, confidence=0.8)

    results = _fitter(monkeypatch, model).fit(image, [box, second])

    assert len(results) == 2


def test_fit_returns_empty_list_when_the_model_reports_no_fit(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray, box: Detection
) -> None:
    """``ok=False`` means the fit did not converge, not that a call failed."""
    model = _FakeFacemark(None, ok=False)

    assert _fitter(monkeypatch, model).fit(image, [box]) == []


def test_model_is_loaded_once_across_calls(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray, box: Detection
) -> None:
    """The 54 MB model loads lazily on first use and is reused afterwards."""
    loads: list[str] = []

    def _fake_load(path: str) -> _FakeFacemark:
        loads.append(path)
        return _FakeFacemark([_indexed_face()])

    monkeypatch.setattr(landmarks_module, "_load_facemark", _fake_load)

    landmarker = FaceLandmarker("models/face/lbfmodel.yaml")
    assert loads == []

    landmarker.fit(image, [box])
    landmarker.fit(image, [box])

    assert loads == ["models/face/lbfmodel.yaml"]


def test_fit_without_boxes_never_loads_the_model(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray
) -> None:
    """No face means no fit, so a frame with nobody in it costs nothing."""

    def _fail(_path: str) -> _FakeFacemark:
        raise AssertionError("the model must not load when there are no faces")

    monkeypatch.setattr(landmarks_module, "_load_facemark", _fail)

    assert FaceLandmarker("models/face/lbfmodel.yaml").fit(image, []) == []


def test_fit_raises_file_not_found_when_the_model_is_missing(
    tmp_path: Path, image: np.ndarray, box: Detection
) -> None:
    """A missing model must not surface as OpenCV's own ``cv2.error``.

    The route maps ``FileNotFoundError`` to 503; ``cv2.error`` would escape as a
    500, so this check is what keeps the documented contract true.
    """
    missing = tmp_path / "absent.yaml"

    with pytest.raises(FileNotFoundError, match=str(missing)):
        FaceLandmarker(str(missing)).fit(image, [box])


def test_mesh_reports_the_jaw_when_requested(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray, box: Detection
) -> None:
    """``mesh=True`` opts into the face boundary the default withholds."""
    model = _FakeFacemark([_indexed_face()])

    (result,) = _fitter(monkeypatch, model).fit(image, [box], mesh=True)

    assert result.jaw == [(index, 100 + index) for index in range(17)]


def test_mesh_reports_all_sixty_eight_points_in_ibug_order(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray, box: Detection
) -> None:
    """``points`` is the flat array the triangle indices refer into.

    Clients could concatenate the seven groups themselves, but that ordering
    would be an invisible contract no test could catch if it drifted.
    """
    model = _FakeFacemark([_indexed_face()])

    (result,) = _fitter(monkeypatch, model).fit(image, [box], mesh=True)

    assert result.points == [(index, 100 + index) for index in range(68)]
    assert result.points is not None
    assert result.points[0:17] == result.jaw
    assert result.points[48:68] == result.mouth


def test_mesh_triangle_indices_stay_inside_the_point_array(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray, box: Detection
) -> None:
    """Subdiv2D's virtual outer rectangle must not leak into the output.

    ``getTriangleList`` returns triangles touching that rectangle; unfiltered,
    their vertices are not landmarks at all and index nothing.
    """
    model = _FakeFacemark([_scattered_face()])

    (result,) = _fitter(monkeypatch, model).fit(image, [box], mesh=True)

    assert result.triangles
    for triangle in result.triangles:
        assert len(set(triangle)) == 3
        assert all(0 <= index < 68 for index in triangle)


def test_mesh_uses_every_landmark(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray, box: Detection
) -> None:
    """Points in general position all take part in the triangulation."""
    model = _FakeFacemark([_scattered_face()])

    (result,) = _fitter(monkeypatch, model).fit(image, [box], mesh=True)

    assert result.triangles is not None
    assert {index for triangle in result.triangles for index in triangle} == set(
        range(68)
    )


def test_mesh_survives_degenerate_points(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray, box: Detection
) -> None:
    """Collinear or coincident fits produce no triangles rather than raising.

    A failed fit can return points that share a coordinate or fall on one line;
    that is a bad frame, not a server error.
    """
    collinear = _FakeFacemark([_indexed_face()])
    coincident = _FakeFacemark([np.full((1, 68, 2), 50.0, dtype=np.float32)])

    (flat,) = _fitter(monkeypatch, collinear).fit(image, [box], mesh=True)
    (stacked,) = _fitter(monkeypatch, coincident).fit(image, [box], mesh=True)

    assert flat.triangles == []
    assert stacked.triangles == []


def test_mesh_accepts_points_outside_the_face_box(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray, box: Detection
) -> None:
    """The triangulation bounds come from the points, not the detection box.

    Fitted points routinely fall outside the box that seeded them, and
    Subdiv2D raises on any insert outside its rectangle.
    """
    points = _scattered_face()
    points[0][0] = (box.x1 - 40.0, box.y1 - 40.0)
    points[0][1] = (box.x2 + 40.0, box.y2 + 40.0)

    (result,) = _fitter(monkeypatch, _FakeFacemark([points])).fit(
        image, [box], mesh=True
    )

    assert result.triangles


def test_fit_leaves_the_mesh_fields_unset_by_default(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray, box: Detection
) -> None:
    """Not asking for the mesh costs nothing and reports nothing extra."""
    model = _FakeFacemark([_scattered_face()])

    (result,) = _fitter(monkeypatch, model).fit(image, [box])

    assert (result.jaw, result.points, result.triangles) == (None, None, None)
