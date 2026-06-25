import pymupdf4llm

# Extract the PDF to a Markdown string
md_text = pymupdf4llm.to_markdown("9789241548281_Vol1_eng.pdf")

# Save it to a file
with open("who_volume1_clean.md", "w", encoding="utf-8") as f:
    f.write(md_text)