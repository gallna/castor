from castor.speakable import to_speakable


def test_plain_text_is_unchanged():
    assert to_speakable("Hello there. How are you?") == "Hello there. How are you?"


def test_bold_and_italic_markers_are_removed():
    assert to_speakable("This is **very** *important* and __bold__.") == (
        "This is very important and bold."
    )


def test_headings_and_bullets_are_flattened():
    text = "# Title\n- first item\n* second item\n1. third item"

    assert to_speakable(text) == "Title first item second item third item"


def test_links_keep_only_their_label():
    assert to_speakable("See [the docs](https://example.com) now.") == "See the docs now."


def test_inline_code_keeps_content_and_code_blocks_are_dropped():
    text = "Run `ls` like this:\n```bash\nls -la\n```\nDone."

    assert to_speakable(text) == "Run ls like this: Done."


def test_whitespace_is_collapsed():
    assert to_speakable("  one\n\n two   three ") == "one two three"
