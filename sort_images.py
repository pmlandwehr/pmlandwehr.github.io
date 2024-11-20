"""A li'l script for generating a sorted table of images."""

import argparse
import csv
from functools import cache, cached_property
from pathlib import Path
from typing import Callable

from PIL import Image

RELATIVE_THUMB = Path("images/thumbnail")
RELATIVE_FULL = Path("images/full")

_known_categories = {
    "art",
    "babadook",
    "blitt",
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
    "new-yorker",
    "perry-bible-fellowship",
    "post",
    "sarahs-scribbles",
    "text",
    "tumblr",
    "twitter",
}

_implied_categories = {
    "black-and-white": {"buttercup-festival", "chainsawsuit", "kelly", "sarahs-scribbles", "xkcd"},
    "comic": {
        "blitt",
        "buttercup-festival",
        "chainsawsuit",
        "kelly",
        "louder-and-smarter",
        "perry-bible-fellowship",
        "sarahs-scribbles",
        "xkcd",
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
            f'<td><a href="{RELATIVE_FULL!s}/{self.name}.png">'
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


@cache
def height_difference(key_one: str, key_two: str) -> int:
    """Return the height difference between two images."""
    return image_dict[key_two].thumbnail.height - image_dict[key_one].thumbnail.height


def extracted_row_of_images(
    remaining_images_in_order: list[str], images_per_row: int, image_filters: list[Callable[[str, str], bool]]
) -> list[str]:
    """Extract the next row of images from the sorted list of remaining images.

    Parameters
    ----------
    remaining_images_in_order : list[str]
        The remaining images to be collected into table rows, highest priority images first.

    images_per_row : int
        The number of images to place in the new row.

    image_filters : list[Callable[[str, str], bool]]
        List of image tests. All images passing the first filter should be evaluated before images passing the
        second filter, etc.

    Returns
    -------
    list[str]
        Row of images

    """
    row = [remaining_images_in_order.pop(0)]

    while len(row) < images_per_row and len(remaining_images_in_order) > 0:
        next_image = remaining_images_in_order[0]

        for image_filter in image_filters:
            candidate_images = []
            for key in remaining_images_in_order:
                if image_filter(row[-1], key):
                    candidate_images.append(key)
                else:
                    break

            if len(candidate_images) < 1:
                continue

            next_image = sorted(candidate_images, key=lambda x: jaccard_similarity(row[-1], x))[0]
            break

        row.append(next_image)
        remaining_images_in_order.remove(next_image)

    return row


def generate_table_interior(
    shortest_to_longest: list[str], images_per_row: int, max_pixel_diff: int
) -> list[list[str | None]]:
    """Generate the interior portions of a table of image names.

    Parameters
    ----------
    shortest_to_longest: list[str]
        Sorted list of image names
    images_per_row : int
        Number of images to put in each row of the table

    Returns
    -------
    list[list[str]]
    """
    counter = 0
    rows = []
    while len(shortest_to_longest) > 0:
        if counter % 2 == 0:
            # Add row that is roughly increasing in height, growing from shortest
            rows.append(
                extracted_row_of_images(
                    shortest_to_longest,
                    images_per_row,
                    [
                        lambda last_key, key: height_difference(last_key, key) < -max_pixel_diff,
                        lambda last_key, key: height_difference(last_key, key) <= max_pixel_diff,
                    ],
                )
            )
        else:
            # Add row that is roughly decreasing in height, shrinking from longest
            longest_to_shortest = list(reversed(shortest_to_longest))
            rows.append(
                extracted_row_of_images(
                    longest_to_shortest,
                    images_per_row,
                    [
                        lambda last_key, key: height_difference(last_key, key) > max_pixel_diff,
                        lambda last_key, key: height_difference(last_key, key) >= -max_pixel_diff,
                    ],
                )
            )
            shortest_to_longest = list(reversed(longest_to_shortest))

        counter += 1

    return rows


def generate_table(images_per_row: int, max_height_difference: int) -> list[list[str | None]]:
    """Generate a table of image names."""
    shortest_to_longest = sorted(
        [key for key in image_dict.keys() if key != "butts"], key=lambda x: image_dict[x].thumbnail.height
    )

    rows = generate_table_interior(shortest_to_longest, images_per_row, max_height_difference)

    if len(rows[-1]) == images_per_row:
        rows.append([None] * (images_per_row - 1) + ["butts"])
    else:
        rows[-1] = rows[-1] + [None] * (images_per_row - 1 - len(rows[-1])) + ["butts"]

    return rows


def table_to_str(rows: list[list[str | None]]) -> str:
    """Convert a table of rows into an HTML table string."""
    html_strings = [8 * " " + "<table>"]
    for row in rows:
        html_strings.append(12 * " " + "<tr>")
        for cell in row:
            html_strings.append(16 * " " + ("<td></td>" if cell is None else image_dict[cell].table_cell()))
        html_strings.append(12 * " " + "</tr>")
    html_strings.append(8 * " " + "</table>")

    return "\n".join(html_strings)


def main() -> None:
    """Entry point."""

    parser = argparse.ArgumentParser(description="Generate a chunk of an HTML table.")
    parser.add_argument("--images_per_row", help="Number of images per row", type=int, default=4)
    parser.add_argument(
        "--max_height_difference", help="Maximum difference in height between images", type=int, default=5
    )
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

    print(table_to_str(generate_table(args.images_per_row, args.max_height_difference)))


if __name__ == "__main__":
    main()
