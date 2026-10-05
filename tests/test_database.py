from app.database.repository import ConversationRepository, DatasetRepository, SessionRepository


def test_session_and_history_order():
    SessionRepository.create("s1", {"a": 1})
    assert SessionRepository.get("s1")["settings"] == {"a": 1}
    for i in range(5):  # bir sekundda yozilgan xabarlar tartibi saqlanishi kerak
        ConversationRepository.add_message("s1", "user", f"m{i}")
    assert [m["content"] for m in ConversationRepository.get_history("s1")] == [f"m{i}" for i in range(5)]
    assert [m["content"] for m in ConversationRepository.get_recent("s1", 2)] == ["m3", "m4"]


def test_cascade_delete():
    SessionRepository.create("s2", {})
    DatasetRepository.register("d1", "s2", "a.csv", "/x/a.csv", "csv", 1, 1, {"a": "int64"})
    SessionRepository.delete("s2")
    assert DatasetRepository.get("d1") is None
