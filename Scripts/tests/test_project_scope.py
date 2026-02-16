from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

SCRIPT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from codex_mem import (
    ensure_session,
    get_item_detail,
    insert_event,
    insert_observation,
    open_db,
    timeline_for_event,
    timeline_for_observation,
)


class ProjectScopeTests(unittest.TestCase):
    def test_project_scope_guard_for_timeline_and_get(self) -> None:
        with tempfile.TemporaryDirectory(prefix="codex_mem_scope_") as tmp:
            root = pathlib.Path(tmp)
            conn = open_db(root, ".codex_mem")

            ensure_session(conn, "sA", "projA", "Session A")
            ensure_session(conn, "sB", "projB", "Session B")

            event_a = insert_event(
                conn,
                session_id="sA",
                project="projA",
                event_kind="tool",
                role="tool",
                title="A event",
                content="A content",
                tool_name="shell",
                file_path="a.txt",
                tags=["a"],
                metadata={},
            )
            event_b = insert_event(
                conn,
                session_id="sB",
                project="projB",
                event_kind="tool",
                role="tool",
                title="B event",
                content="B content",
                tool_name="shell",
                file_path="b.txt",
                tags=["b"],
                metadata={},
            )
            obs_b = insert_observation(
                conn,
                session_id="sB",
                project="projB",
                observation_type="learning",
                title="B observation",
                body="B body",
                source_event_ids=[event_b],
                metadata={},
            )
            conn.commit()

            detail_ok = get_item_detail(conn, "event", event_a, 240, project="projA", include_private=False)
            self.assertEqual(detail_ok.get("project"), "projA")

            with self.assertRaisesRegex(ValueError, "not in project 'projA'"):
                get_item_detail(conn, "event", event_b, 240, project="projA", include_private=False)

            with self.assertRaisesRegex(ValueError, "not in project 'projA'"):
                timeline_for_event(
                    conn,
                    event_id=event_b,
                    project="projA",
                    before=2,
                    after=2,
                    snippet_chars=240,
                    include_private=False,
                )

            with self.assertRaisesRegex(ValueError, "not in project 'projA'"):
                timeline_for_observation(
                    conn,
                    obs_id=obs_b,
                    project="projA",
                    before=2,
                    after=2,
                    snippet_chars=240,
                    include_private=False,
                )

            # Sanity: matching project still works.
            timeline_ok = timeline_for_event(
                conn,
                event_id=event_a,
                project="projA",
                before=2,
                after=2,
                snippet_chars=240,
                include_private=False,
            )
            self.assertEqual(timeline_ok.get("anchor", {}).get("project"), "projA")


if __name__ == "__main__":
    unittest.main()
