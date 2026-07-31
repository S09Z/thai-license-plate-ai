"""Deterministic mapping of a recognized province candidate to a real province."""

import re

from postprocess.thai import strip_thai_marks

THAI_PROVINCES: tuple[str, ...] = (
    "กรุงเทพมหานคร",
    "กระบี่",
    "กาญจนบุรี",
    "กาฬสินธุ์",
    "กำแพงเพชร",
    "ขอนแก่น",
    "จันทบุรี",
    "ฉะเชิงเทรา",
    "ชลบุรี",
    "ชัยนาท",
    "ชัยภูมิ",
    "ชุมพร",
    "เชียงราย",
    "เชียงใหม่",
    "ตรัง",
    "ตราด",
    "ตาก",
    "นครนายก",
    "นครปฐม",
    "นครพนม",
    "นครราชสีมา",
    "นครศรีธรรมราช",
    "นครสวรรค์",
    "นนทบุรี",
    "นราธิวาส",
    "น่าน",
    "บึงกาฬ",
    "บุรีรัมย์",
    "ปทุมธานี",
    "ประจวบคีรีขันธ์",
    "ปราจีนบุรี",
    "ปัตตานี",
    "พระนครศรีอยุธยา",
    "พะเยา",
    "พังงา",
    "พัทลุง",
    "พิจิตร",
    "พิษณุโลก",
    "เพชรบุรี",
    "เพชรบูรณ์",
    "แพร่",
    "ภูเก็ต",
    "มหาสารคาม",
    "มุกดาหาร",
    "แม่ฮ่องสอน",
    "ยโสธร",
    "ยะลา",
    "ร้อยเอ็ด",
    "ระนอง",
    "ระยอง",
    "ราชบุรี",
    "ลพบุรี",
    "ลำปาง",
    "ลำพูน",
    "เลย",
    "ศรีสะเกษ",
    "สกลนคร",
    "สงขลา",
    "สตูล",
    "สมุทรปราการ",
    "สมุทรสงคราม",
    "สมุทรสาคร",
    "สระแก้ว",
    "สระบุรี",
    "สิงห์บุรี",
    "สุโขทัย",
    "สุพรรณบุรี",
    "สุราษฎร์ธานี",
    "สุรินทร์",
    "หนองคาย",
    "หนองบัวลำภู",
    "อำนาจเจริญ",
    "อุดรธานี",
    "อุตรดิตถ์",
    "อุทัยธานี",
    "อุบลราชธานี",
    "อ่างทอง",
)

_WHITESPACE_PATTERN = re.compile(r"\s+")


def _lookup_key(text: str) -> str:
    """Reduce province text to the form used for comparison.

    Args:
        text: A province name or a recognized candidate.

    Returns:
        The text without whitespace or Thai combining marks.
    """
    return strip_thai_marks(_WHITESPACE_PATTERN.sub("", text))


# Built once at import: the mark-free skeleton of each province maps back to
# its canonical spelling. A test asserts no two provinces share a skeleton.
_BY_SKELETON: dict[str, str] = {
    _lookup_key(province): province for province in THAI_PROVINCES
}


def match_province(candidate: str) -> str | None:
    """Map a recognized province candidate onto a canonical province name.

    Matching ignores whitespace and Thai combining marks, because the
    recognizer drops vowel and tone marks while reading consonants reliably.
    It is otherwise exact: a candidate that is merely *close* to a province is
    rejected rather than guessed at, leaving Phase 5 (RAG) to resolve it.

    Args:
        candidate: A province candidate, exactly as recognized.

    Returns:
        The canonical province name, or ``None`` when the candidate does not
        match one.
    """
    return _BY_SKELETON.get(_lookup_key(candidate))
