# -*- coding: utf-8 -*-
"""中文查询归一化：繁体转简体、常见错字纠正、口语化时间归一。

仅作用于**查询侧**（rag_pipeline.normalize_query），不改动入库文档：
错字纠正与繁简转换若应用到文档会污染原文且无法回退；查询侧归一化则能在
检索前把「繁转简 + 纠错 + 时间具体化」做好，提升中文场景召回。
"""
from __future__ import annotations

import calendar
from datetime import datetime, timedelta

# ----------------------------------------------------------------------
#  繁体 -> 简体（单字映射，覆盖常见繁体字；无第三方依赖）
# ----------------------------------------------------------------------
_T2S = {
    "請": "请", "問": "问", "這": "这", "個": "个", "們": "们", "嗎": "吗",
    "麼": "么", "說": "说", "話": "话", "與": "与", "對": "对", "為": "为",
    "會": "会", "開": "开", "關": "关", "間": "间", "見": "见", "聽": "听",
    "報": "报", "銷": "销", "費": "费", "單": "单", "資": "资", "產": "产",
    "業": "业", "應": "应", "該": "该", "還": "还", "來": "来", "過": "过",
    "學": "学", "師": "师", "員": "员", "電": "电", "腦": "脑", "視": "视",
    "頻": "频", "訊": "讯", "係": "系", "統": "统", "設": "设", "計": "计",
    "劃": "划", "歸": "归", "類": "类", "檔": "档", "規": "规", "則": "则",
    "製": "制", "審": "审", "發": "发", "戶": "户", "碼": "码", "網": "网",
    "絡": "络", "頁": "页", "數": "数", "據": "据", "庫": "库", "體": "体",
    "臺": "台", "灣": "湾", "從": "从", "無": "无", "時": "时", "處": "处",
    "進": "进", "項": "项", "簽": "签", "約": "约", "證": "证", "務": "务",
    "財": "财", "稅": "税", "補": "补", "貼": "贴", "車": "车", "門": "门",
    "長": "长", "團": "团", "隊": "队", "專": "专", "讓": "让", "講": "讲",
    "讀": "读", "寫": "写", "認": "认", "識": "识",
    "訴": "诉", "試": "试", "詢": "询", "聯": "联", "續": "续",
    "議": "议", "論": "论", "討": "讨", "質": "质", "標": "标",
    "準": "准", "備": "备", "預": "预", "決": "决", "執": "执",
    "現": "现", "經": "经", "營": "营", "買": "买", "賣": "卖",
    "購": "购", "價": "价", "錢": "钱", "銀": "银", "額": "额",
    "張": "张", "條": "条", "帳": "账", "匯": "汇", "轉": "转",
    "選": "选", "擇": "择", "記": "记", "錄": "录", "協": "协",
    "詳": "详", "細": "细", "萬": "万", "億": "亿",
    "幫": "帮", "週": "周", "樣": "样", "邊": "边", "裡": "里",
    "後": "后", "髮": "发", "幹": "干", "幾": "几", "點": "点",
    "頭": "头", "實": "实", "際": "际", "滿": "满", "驗": "验",
    "題": "题", "況": "况", "態": "态", "變": "变", "難": "难",
}
_T2S_TABLE = str.maketrans(_T2S)


def to_simplified(text: str) -> str:
    """繁体 -> 简体（字符级映射）。"""
    if not text:
        return text
    return text.translate(_T2S_TABLE)


# ----------------------------------------------------------------------
#  常见同音/形近错字纠正（保守白名单，只替换极明确的错误）
# ----------------------------------------------------------------------
_TYPO_MAP = {
    "报消": "报销",
    "合通": "合同",
    "员功": "员工",
    "部们": "部门",
    "社宝": "社保",
}


def correct_typos(text: str) -> str:
    """纠正少量高频同音/形近错字（仅查询侧，保守处理）。"""
    if not text:
        return text
    for wrong, right in _TYPO_MAP.items():
        if wrong in text:
            text = text.replace(wrong, right)
    return text


# ----------------------------------------------------------------------
#  口语化时间归一
# ----------------------------------------------------------------------
_WEEKDAY_NAMES = "一二三四五六日"
_WEEKDAY_IDX = {ch: i for i, ch in enumerate(_WEEKDAY_NAMES)}


def _weekday_date(now: datetime, target_idx: int, week_offset: int) -> datetime:
    """求 week_offset（-1 上周 / 0 本周 / 1 下周）中某星期几的日期。"""
    monday = now - timedelta(days=now.weekday())
    return monday + timedelta(days=target_idx + week_offset * 7)


def _shift_month(now: datetime, delta: int) -> datetime:
    y = now.year
    m = now.month + delta
    while m < 1:
        m += 12
        y -= 1
    while m > 12:
        m -= 12
        y += 1
    day = min(now.day, calendar.monthrange(y, m)[1])
    return now.replace(year=y, month=m, day=day)


def normalize_time_expressions(text: str, now: datetime | None = None) -> str:
    """把口语化相对时间转成具体日期，便于检索带时效的条文。"""
    if not text:
        return text
    now = now or datetime.now()

    # 相对年（"今年上半年"拆成"2026年"+"上半年"即可）
    for word, year in (
        ("前年", now.year - 2),
        ("去年", now.year - 1),
        ("今年", now.year),
        ("明年", now.year + 1),
    ):
        text = text.replace(word, f"{year}年")

    # 相对月（"上个月"必须先于"上月"替换）
    for word, delta in (
        ("上个月", -1), ("这个月", 0), ("下个月", 1),
        ("上月", -1), ("本月", 0), ("下月", 1),
    ):
        m = _shift_month(now, delta)
        text = text.replace(word, f"{m.year}年{m.month}月")

    # 相对日
    for word, delta in (
        ("前天", -2), ("昨天", -1), ("今天", 0), ("明天", 1), ("后天", 2),
    ):
        d = now + timedelta(days=delta)
        text = text.replace(word, f"{d.year}年{d.month}月{d.day}日")

    # 上周/本周/下周 + 星期几（如"上周五"）
    for prefix, offset in (("上周", -1), ("本周", 0), ("这周", 0), ("下周", 1)):
        for ch in _WEEKDAY_NAMES:
            d = _weekday_date(now, _WEEKDAY_IDX[ch], offset)
            text = text.replace(prefix + ch, f"{d.year}年{d.month}月{d.day}日")

    # 星期X / 周X（按本周理解）
    for ch in _WEEKDAY_NAMES:
        d = _weekday_date(now, _WEEKDAY_IDX[ch], 0)
        text = text.replace("星期" + ch, f"{d.year}年{d.month}月{d.day}日")
        text = text.replace("周" + ch, f"{d.year}年{d.month}月{d.day}日")

    return text


def normalize_cjk(query: str, now: datetime | None = None) -> str:
    """查询侧中文归一化：繁体转简体 -> 错字纠正 -> 口语化时间具体化。"""
    if not query:
        return query
    query = to_simplified(query)
    query = correct_typos(query)
    query = normalize_time_expressions(query, now=now)
    return query
