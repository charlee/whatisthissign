"""Extract images using annotation and convert to PNG format."""

import os
import csv
import argparse
from PIL import Image


IMAGE_SIZE = 64


def extract_images(index, output_dir):
    """Extract images annotated by index (a CSV file) from relative_path
    and save the extracted images to output_dir."""

    dir = os.path.dirname(index)

    # Extract necessary info from the csv index
    output_entries = []

    with open(index, 'r') as f:
        csv_reader = csv.reader(f, delimiter=';')
        for (filename, width, height, x1, y1, x2, y2, class_id) in csv_reader:
            if filename == 'Filename':
                continue

            image_filename = os.path.join(dir, filename)
            if os.path.isfile(image_filename):
                im = Image.open(image_filename)
                im = im.convert('RGB')
                im = im.crop((int(x1), int(y1), int(x2), int(y2)))

                basename = os.path.splitext(os.path.basename(image_filename))[0]
                output_filename = '{}.png'.format(basename)

                im.save(os.path.join(output_dir, output_filename))
                im.close()

                output_entries.append([output_filename, class_id])

    output_csv_file = os.path.join(output_dir, os.path.basename(index))
    with open(output_csv_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        for row in output_entries:
            csv_writer.writerow(row)

    return len(output_entries)


def main(image_dir, output_dir):
    """Traverse image_dir and handle with each CSV annotations."""

    dir_count = 0
    for root, dirs, files in os.walk(image_dir):
        relative_path = root[len(image_dir)+1:]
        for file in files:
            if file.endswith('.csv'):
                # Create output dir
                output_path = os.path.join(output_dir, relative_path)
                if not os.path.isdir(output_path):
                    os.makedirs(output_path)

                file_count = extract_images(os.path.join(root, file), output_path)
                dir_count += 1

                print('{}: {} files converted.'.format(root, file_count))

    print('Total {} dirs converted.'.format(dir_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Image directory.',
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for cropped images.',
    )

    flags, unparsed = parser.parse_known_args()
    main(image_dir=flags.image_dir, output_dir=flags.output_dir)
