"""A li'l script for generating a sorted table of images."""

import argparse
import csv
from functools import cache, cached_property
import operator
from pathlib import Path

from PIL import Image

RELATIVE_THUMB = Path("images/thumbnail")
RELATIVE_FULL = Path("images/full")

_known_categories = {
    "art",
    "buttercup-festival",
    "chainsawsuit",
    "color",
    "comic",
    "dril",
    "image",
    "kelly",
    "louder-and-smarter",
    "mostly-black",
    "mostly-white",
    "naomi-wolf",
    "perry-bible-fellowship",
    "post",
    "sarahs-scribbles",
    "text",
    "tumblr",
    "twitter",
}

_implied_categories = {
    "black-and-white": {"buttercup-festival", "chainsawsuit", "kelly", "sarahs-scribbles"},
    "comic": {
        "buttercup-festival",
        "chainsawsuit",
        "kelly",
        "louder-and-smarter",
        "perry-bible-fellowship",
        "sarahs-scribbles",
    },
    "image": {"comic"},
    "post": {"twitter", "tumblr"},
    "text": {"dril"},
    "twitter": {"dril", "naomi-wolf"},
}

_exclusive_categories = [
    {"color", "black-and-white"},
    {"mostly-black", "mostly-white"},
    {"image", "text"},
    {"buttercup-festival", "chainsawsuit", "kelly", "louder-and-smarter", "perry-bible-fellowship", "sarahs-scribbles"},
    {"dril", "naomi-wolf"},
    {"twitter", "tumblr"},
]


class ImageData:
    """Wrapper for metadata about the different image files."""

    def __init__(self, name: str, alt_text: str, categories: set[str]):
        self._name = name
        self._alt_text = alt_text
        self._categories = categories

        while self.add_implied_categories():
            pass

        if "post" in self.categories and "image" not in self.categories:
            self._categories.add("text")

        if "image" in self.categories and "black-and-white" not in self.categories:
            self._categories.add("color")

        for exclusive_category_set in _exclusive_categories:
            common = self.categories & exclusive_category_set
            if len(common) > 1:
                raise ValueError(common)

        for path in self.full_path, self.thumbnail_path:
            if not path.is_file():
                raise FileNotFoundError(path)

    def add_implied_categories(self) -> bool:
        """Try to add implied categories, returning True if successful."""
        success = False
        for implied_category, source_categories in _implied_categories.items():
            if implied_category in self.categories:
                continue
            if any(source_category in self.categories for source_category in source_categories):
                self._categories.add(implied_category)
                success = True
        return success

    @cached_property
    def thumbnail(self):
        return Image.open(self.thumbnail_path)

    @property
    def name(self) -> str:
        return self._name

    @property
    def alt_text(self) -> str:
        return self._alt_text

    @property
    def categories(self) -> set[str]:
        return self._categories

    @property
    def full_path(self):
        """Return the path to the thumbnail image."""
        return RELATIVE_FULL / (self.name + ".png")

    @property
    def thumbnail_path(self):
        """Return the path to the thumbnail image."""
        return RELATIVE_THUMB / (self.name + ".png")

    def table_cell(self) -> str:
        return (
            f'<td><a href="{RELATIVE_FULL!s}/{self.name}.png">"'
            f'<img src="{RELATIVE_THUMB!s}/{self.name}.png" alt="{self.alt_text}">'
            "</a></td>"
        )

with open("metadata.csv") as infile:
    image_dict = {
        row["name"]: ImageData(row["name"], row["alt_text"], set(row["categories"].split(",")))
        for row in csv.DictReader(infile)
    }


@cache
def jaccard_similarity(key_one: str, key_two: str) -> float:
    """Calculate the Jaccard similarity of two images based on metadata categories."""
    if key_one > key_two:
        return jaccard_similarity(key_two, key_one)
    first_categories = image_dict[key_one].categories
    second_categories = image_dict[key_two].categories
    return len(first_categories & second_categories) / len(first_categories | second_categories)


# Recurse:
# find largest or smallest
# find next X images within y pixels (incl. all larger/smaller)
# find furthest away image
# repeat.

def row_getting_smaller(shortest_to_longest: list[str], images_per_row: int, max_pixel_diff: int) -> list[str]:
    row = [shortest_to_longest.pop(-1)]

    while len(row) < images_per_row and len(shortest_to_longest) > 0:
        candidate_images = []
        for key in reversed(shortest_to_longest):
            delta = image_dict[key].thumbnail.height - image_dict[row[-1]].thumbnail.height

            if operator.le(delta, -max_pixel_diff):
                candidate_images.append((0, key))
                break
            if operator.le(delta, max_pixel_diff):
                candidate_images.append((jaccard_similarity(row[-1], key), key))
            else:
                break

        if len(candidate_images) < 1:
            row.append(shortest_to_longest.pop(-1))
        else:
            next_image = sorted(candidate_images)[0][1]
            row.append(next_image)
            shortest_to_longest.remove(next_image)

    return row


def row_getting_larger(shortest_to_longest: list[str], images_per_row: int, max_height_delta: int) -> list[str]:
    row = [shortest_to_longest.pop(0)]

    while len(row) < images_per_row and len(shortest_to_longest) > 0:

        candidate_images = []
        for key in shortest_to_longest:
            delta = image_dict[key].thumbnail.height - image_dict[row[-1]].thumbnail.height
            if operator.ge(delta, -max_height_delta):
                candidate_images.append((0, key))
                break
            if operator.ge(delta, max_height_delta):
                candidate_images.append((jaccard_similarity(row[-1], key), key))
            else:
                break

        if len(candidate_images) < 1:
            row.append(shortest_to_longest.pop(0))
        else:
            next_image = sorted(candidate_images)[0][1]
            row.append(next_image)
            shortest_to_longest.remove(next_image)

    return row


def generate_table_interior_v2(shortest_to_longest: list[str], images_per_row: int, max_pixel_diff: int) -> list[list[str|None]]:
    """Generate the interior portions of a table of image names.

    Parameters
    ----------
    shortest_to_longest: list[str]
        Sorted list of image names
    images_per_row : int
        Number of images to put in each row of the table

    Returns
    -------
    list[list[str|None]]
    """
    counter = 0
    rows = []
    while len(shortest_to_longest) > 0:
        rows.append((row_getting_larger if counter % 2 == 0 else row_getting_smaller)(shortest_to_longest, images_per_row, max_pixel_diff))
        counter += 1

    return rows


def generate_table_interior(shortest_to_longest: list[str], images_per_row: int, _: int) -> list[list[str|None]]:
    """Generate the interior portions of a table of image names.

    Parameters
    ----------
    images_per_row : int
        Number of images to put in each row of the table

    Returns
    -------
    list[list[str|None]]
    """
    rows = []
    for left_of_left_ix, right_of_right_ix in zip(
        range(0, len(shortest_to_longest), images_per_row), range(len(shortest_to_longest) - 1, -1, -images_per_row)
    ):
        if left_of_left_ix > right_of_right_ix:
            break

        right_of_left_ix = min(left_of_left_ix + images_per_row, right_of_right_ix)
        left_of_right_ix = max(right_of_right_ix - images_per_row, right_of_left_ix)

        for slicer in slice(left_of_left_ix, right_of_left_ix), slice(right_of_right_ix, left_of_right_ix, -1):
            cur_slice = shortest_to_longest[slicer]
            if len(cur_slice) > 0:
                rows.append(cur_slice)

    return rows


def generate_table(images_per_row: int, max_height_delta: int) -> list[list[str|None]]:
    """Generate a table of image names."""
    shortest_to_longest = sorted(
        [key for key in image_dict.keys() if key != "butts"], key=lambda x: image_dict[x].thumbnail.height
    )

    rows = generate_table_interior(shortest_to_longest, images_per_row)

    if len(rows[-1]) == images_per_row:
        rows.append([None] * (images_per_row -1) + ["butts"])
    else:
        rows[-1] = rows[-1] + [None] * (images_per_row - 1 - len(rows[-1])) + ["butts"]

    return rows


def table_to_str(rows: list[list[str|None]]) -> str:
    """Convert a table of rows into an HTML table string."""
    html_strings = [8 * " " + "<table>"]
    for row in rows:
        html_strings.append(12 * " " + "<tr>")
        for cell in row:
            html_strings.append(16 * " " +  ("<td></td>" if cell is None else image_dict[cell].table_cell()))
        html_strings.append(12 * " " + "</tr>")
    html_strings.append(8 * " " + "</table>")

    return "\n".join(html_strings)


def main() -> None:
    """Entry point."""

    parser = argparse.ArgumentParser(description="Generate a chunk of an HTML table.")
    parser.add_argument("--images_per_row", help="Number of images per row", type=int, default=4)
    parser.add_argument("--max_height_delta", help="Maximum difference in height between images", type=int, default=5)
    args = parser.parse_args()

    for path in RELATIVE_THUMB, RELATIVE_FULL:
        if not path.exists():
            raise FileNotFoundError(path)

    thumbnail_png_paths = RELATIVE_THUMB.glob("*")
    full_png_paths = RELATIVE_FULL.glob("*")

    thumbnail_names = {path.stem for path in thumbnail_png_paths}
    full_names = {path.stem for path in full_png_paths}
    image_dict_names = set(image_dict.keys())
    missing_image_dict_names = set()

    for name, image_names in [("thumbnail", thumbnail_names), ("full", full_names)]:
        for image_name in image_names - image_dict_names:
            print(f"Missing {name} metadata: {image_name}")
        for image_name in image_dict_names - image_names:
            print(f"Missing {name} file: {image_name}")
            missing_image_dict_names.add(image_name)

    for key in missing_image_dict_names:
        del image_dict[key]

    print(table_to_str(generate_table(args.images_per_row, args.max_height_delta)))


if __name__ == "__main__":
    main()
