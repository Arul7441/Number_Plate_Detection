from pathlib import Path

label_dirs = [
    Path("dataset/labels/train"),
    Path("dataset/labels/val")
]

for label_dir in label_dirs:

    for file in label_dir.glob("*.txt"):

        lines = file.read_text().strip().splitlines()

        if not lines:
            continue

        best_line = None
        best_area = 0

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                continue

            _, x, y, w, h = map(float, parts)

            area = w * h

            if area > best_area:
                best_area = area
                best_line = line

        if best_line:
            file.write_text(best_line + "\n")

print("Labels cleaned successfully.")