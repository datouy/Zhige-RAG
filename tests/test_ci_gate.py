"""CI 评估门禁的行为约定 —— 回归测试。

背景：CI 里的评估门禁此前是这样写的::

    python scripts/evaluate.py ... || echo "::warning::eval gate未通过..."

命令后面接了 `|| echo`，于是**无论门禁是否通过，这一步永远是绿的**。
评估指标退化几个月都不会有人知道 —— 一个"门禁"如果不会让流水线变红，
那它就不是门禁。

现在 CI 直接依赖 evaluate.py 的退出码，约定是：

- 评估集为空（空库冷启，属预期）→ 返回 0，并打印告警；
- **指标不达标** → 返回 1，让流水线变红；
- 达标 → 返回 0。

这里把这三条约定钉住，防止以后有人"顺手"又加回 `|| echo` 或改坏退出码。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run_eval(*extra_args: str, dataset: str = "data/eval/__does_not_exist__.jsonl"):
    """跑 evaluate.py 并返回 (退出码, stdout+stderr)。"""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate.py"),
        "--dataset",
        dataset,
        "--retrieval-only",
        "--no-archive",
        *extra_args,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    return proc.returncode, proc.stdout + proc.stderr


class TestEmptyEvalSet:
    def test_empty_dataset_fails_without_allow_flag(self):
        """默认行为：评估集为空就是失败（不能悄悄放过）。"""
        code, _ = _run_eval()
        assert code == 1, "评估集为空且未显式允许时必须返回非零退出码"

    def test_empty_dataset_passes_with_allow_flag(self):
        """显式 --allow-empty-eval：空库冷启属预期，返回 0。"""
        code, out = _run_eval("--allow-empty-eval")
        assert code == 0, "带 --allow-empty-eval 时空评估集应返回 0"
        assert "warning" in out.lower(), "应给出告警提示，说明本次没有质量结论"


class TestThresholdGateIsNotNeutralized:
    def test_ci_yml_does_not_swallow_eval_failure(self):
        """CI 里不能出现 `|| echo` 把门禁失败吞掉。

        只检查实际的 shell 命令行 —— 注释里为了说明"为什么去掉它"而提到
        同样的字样，不该被误判。
        """
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        command_lines = [
            ln for ln in ci.splitlines() if ln.strip() and not ln.strip().startswith("#")
        ]
        body = "\n".join(command_lines)
        assert "|| echo" not in body, (
            "CI 里 evaluate.py 后面跟了 `|| echo`，评估门禁永远不会让流水线变红"
        )
        assert "continue-on-error" not in ci.split("Eval regression gate")[-1].split("- name:")[0], (
            "评估门禁步骤不应带 continue-on-error"
        )

    def test_ci_has_timeout_to_avoid_infinite_hang(self):
        """CI 必须有超时：挂死的测试不该把流水线挂到 GitHub 的 6 小时上限。"""
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert "timeout-minutes:" in ci, "CI job 缺少 timeout-minutes"

    def test_ci_lints_tests_and_scripts_too(self):
        """lint 范围要覆盖 tests/ 与 scripts/，否则那里的问题永远发现不了。"""
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        ruff_line = next(
            (ln for ln in ci.splitlines() if "ruff check" in ln),
            "",
        )
        assert "tests/" in ruff_line and "scripts/" in ruff_line, (
            f"ruff 检查范围没覆盖 tests/ 与 scripts/：{ruff_line!r}"
        )

    def test_ci_installs_document_parsing_deps(self):
        """CI 必须装文档解析依赖，否则 docx/pdf 解析用例全被 importorskip 跳过。"""
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert "requirements-parse-test.txt" in ci, (
            "CI 未安装文档解析依赖，文档解析路径实际上没有测试覆盖"
        )

    def test_parse_requirements_exist(self):
        req = ROOT / "requirements-parse-test.txt"
        assert req.exists(), (
            "缺少 requirements-parse-test.txt —— 不装它，docx/pdf 解析用例"
            "会被 importorskip 跳过，文档解析路径在 CI 里完全没有覆盖"
        )

    def test_parse_requirements_avoid_heavy_deps(self):
        """别把 torch / streamlit 拖进 CI。

        只取每行的**包名部分**（第一个字段），行内注释不计 ——
        例如 pdfplumber 那行的版本约束注释里提到 streamlit-compatible，
        那是解释版本选择，不等于装了这个重依赖。
        """
        text = (ROOT / "requirements-parse-test.txt").read_text(encoding="utf-8")
        names = []
        for ln in text.splitlines():
            ln = ln.split("#", 1)[0].strip()
            if ln:
                names.append(ln.split()[0].lower())
        for heavy in ("torch", "flagembedding", "sentence-transformers", "streamlit"):
            assert not any(n.startswith(heavy) for n in names), (
                f"requirements-parse-test.txt 不该包含 {heavy}（CI 装它太重）"
            )

    def test_ci_cancels_stale_runs(self):
        """同一分支的新推送应取消旧流水线，避免占用并发额度。"""
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert "concurrency:" in ci, "CI 缺少 concurrency 配置"
        assert "cancel-in-progress" in ci, "CI 未配置 cancel-in-progress"


class TestTopKOverrideDoesNotWriteDeadConfig:
    def test_top_k_only_writes_retrieval_section(self):
        """--top-k 不应再写已删除的 rag.top_k（extra=forbid 会导致校验失败）。"""
        src = (ROOT / "scripts" / "evaluate.py").read_text(encoding="utf-8")
        assert 'cfg.setdefault("rag", {})["top_k"]' not in src, (
            "evaluate.py 仍在写 rag.top_k —— 该字段已从 config.yaml 删除"
        )
        assert 'cfg.setdefault("retrieval", {})["top_k"]' in src
