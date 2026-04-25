import fitz
from PIL import Image
import io

doc = fitz.open("rechnik_bg_zhestov_ezik.pdf")
page = doc[35]  # page 2, 0-indexed



print("=== IMAGES ===")
seen = set()
for img_info in page.get_images(full=True):
    xref = img_info[0]
    if xref in seen: continue
    seen.add(xref)
    rects = page.get_image_rects(xref)
    if rects and rects[0].width > 100:
        r = rects[0]
        print(f"  xref={xref}  cx={((r.x0+r.x1)/2):.0f}  y={r.y0:.0f}-{r.y1:.0f}  w={r.width:.0f}")

print("\n=== TEXT BLOCKS ===")
for b in page.get_text("blocks"):
    x0,y0,x1,y1,text,_,btype = b
    if btype==0 and text.strip():
        cx = (x0+x1)/2
        print(f"  cx={cx:.0f}  y={y0:.0f}-{y1:.0f}  → {repr(text.strip()[:80])}")

print("\n==== FIRST IMAGE DEBUG====")

for img_info in page.get_images(full=True):
    xref = img_info[0]
    rects = page.get_image_rects(xref)
    if not rects or rects[0].width < 100:
        continue
    img_dict = doc.extract_image(xref)
    raw = img_dict["image"]
    cs_name = img_dict.get("cs-name", "")
    
    img = Image.open(io.BytesIO(raw))
    gray = img.convert("L")
    avg = sum(gray.getdata()) / (gray.width * gray.height)
    
    print(f"xref={xref}  cs-name={cs_name[:40]}")
    print(f"  PIL mode={img.mode}  avg_brightness={avg:.1f}")
