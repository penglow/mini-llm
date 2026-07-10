"""
Clean a WhatsApp `_chat.txt` export into a training corpus for a small LM.

Output: data/corpus.txt  ->  one line per message, "name: message"
Consecutive messages are kept in order so the model learns turn-taking and
each person's style. Run:  python prepare_data.py
"""
import os
import re
import unicodedata
from collections import Counter

SRC = "_chat.txt"
OUT_DIR = "data"
OUT = os.path.join(OUT_DIR, "corpus.txt")

# Header of a new message: [dd/mm/yyyy, h:mm:ss AM] Sender: body
HEADER = re.compile(
    r"^\[(\d{2}/\d{2}/\d{4}), ([^\]]+)\] ([^:]+): (.*)$"
)

# Invisible marks WhatsApp injects (LRM, RLM, mention brackets, etc.)
INVISIBLE = dict.fromkeys(map(ord, "‎‏⁦⁧⁨⁩‪‫‬"), None)

# Media / attachment placeholders to strip out of message bodies.
MEDIA = [
    "image omitted", "video omitted", "sticker omitted", "audio omitted",
    "GIF omitted", "document omitted", "Contact card omitted",
    "<Media omitted>", "This message was edited", "<This message was edited>",
]

# Whole messages that are pure system noise -> dropped.
SYSTEM_SUBSTR = [
    "Messages and calls are end-to-end encrypted",
    "created this group", "added you", "changed this group's",
    "changed the group", "changed their phone number", "changed to",
    "You're now an admin", "now an admin", "pinned a message",
    "turned on admin approval", "turned off admin approval",
    "This message was deleted", "You deleted this message",
    "started a call", "missed voice call", "missed video call",
]
# System lines that are exactly "<Name> <verb> ..." for join/leave/admin.
SYSTEM_VERB = re.compile(
    r"\b(added|removed|left|joined|changed this group|became an admin|"
    r"is now an admin|no longer an admin|changed the subject|changed the group)\b"
)

# Senders that are bots / system accounts, not group members.
DROP_SENDERS = {"Meta AI", "Bun School"}

URL = re.compile(r"https?://\S+")


def clean_name(raw: str) -> str:
    """Strip emoji/marks from a display name, keep it stable and readable."""
    raw = raw.translate(INVISIBLE)
    out = []
    for ch in raw:
        cat = unicodedata.category(ch)
        # keep letters, numbers, spaces and a few name punctuation chars
        if cat[0] in ("L", "N") or ch in " /_.-'":
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def strip_media(body: str) -> str:
    for tok in MEDIA:
        body = body.replace(tok, " ")
    return body


def is_system(sender: str, body: str) -> bool:
    if any(s in body for s in SYSTEM_SUBSTR):
        return True
    # A body that is ONLY a system-verb event (no real chat) — these come in
    # with an empty-ish sender or as "<name> added <name>".
    if body.strip() == "" and SYSTEM_VERB.search(sender or ""):
        return True
    return False


def main():
    if not os.path.exists(SRC):
        raise SystemExit(f"Cannot find {SRC} in {os.getcwd()}")

    raw = open(SRC, encoding="utf-8").read()
    lines = raw.split("\n")

    records = []          # (name, body)
    counts = Counter()
    dropped = Counter()

    cur_name = None
    cur_body = []

    def flush():
        nonlocal cur_name, cur_body
        if cur_name is None:
            return
        body = "\n".join(cur_body)
        emit(cur_name, body)
        cur_name, cur_body = None, []

    def emit(name, body):
        name = clean_name(name)
        body = body.translate(INVISIBLE)
        # skip poll blocks
        if body.lstrip().startswith("POLL:") or body.lstrip().startswith("OPTION:"):
            dropped["poll"] += 1
            return
        if name in DROP_SENDERS:
            dropped["bot/system sender"] += 1
            return
        if is_system(name, body):
            dropped["system"] += 1
            return
        body = strip_media(body)
        body = URL.sub("", body)                 # drop links
        body = body.replace("<This message was edited>", "")
        # collapse whitespace / newlines inside a single message
        body = re.sub(r"\s+", " ", body).strip()
        if not body:
            dropped["empty after clean"] += 1
            return
        if not name:
            dropped["no name"] += 1
            return
        records.append((name, body))
        counts[name] += 1

    for line in lines:
        # WhatsApp prefixes many media/edited headers with invisible marks
        # BEFORE the '[', which breaks a '^\[' match — strip marks first.
        line = line.translate(INVISIBLE)
        m = HEADER.match(line)
        if m:
            flush()
            cur_name = m.group(3)
            cur_body = [m.group(4)]
        else:
            # continuation of the previous message (multi-line text)
            if cur_name is not None:
                cur_body.append(line)

    flush()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for name, body in records:
            f.write(f"{name}: {body}\n")

    chars = sum(len(f"{n}: {b}\n") for n, b in records)
    print(f"Wrote {OUT}")
    print(f"  messages kept : {len(records):,}")
    print(f"  characters    : {chars:,}")
    print(f"  speakers      : {len(counts)}")
    print("  per speaker   :")
    for n, c in counts.most_common():
        print(f"      {c:6,}  {n}")
    print("  dropped       :")
    for k, c in dropped.most_common():
        print(f"      {c:6,}  {k}")


if __name__ == "__main__":
    main()
