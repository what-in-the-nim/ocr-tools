from st_pages import Page, Section, show_pages

# Specify what pages should be shown in the sidebar, and what their titles and icons
# should be
show_pages(
    [
        Section("PDF to VIA", icon="📄"),
        Page("pages/convert_pdf_to_image.py", " Convert PDF to Images", "1️⃣"),
        Page("pages/batching_page.py", " Split Images Into Batches", "2️⃣"),
        Page("pages/via_page.py", " Create VIA File", "3️⃣"),
        Section("Tools", icon="🛠️"),
        Page("pages/tool_page.py", "Project Tools", "🛠️"),
    ]
)
