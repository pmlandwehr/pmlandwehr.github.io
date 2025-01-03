"""A li'l script for generating a sorted table of images."""

import argparse
import csv
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import Callable

from PIL import Image

RELATIVE_THUMB = Path("images/thumbnail")
RELATIVE_FULL = Path("images/full")

_known_categories = {
    "art",
    "babadook",
    "new-yorker",
}

_implied_categories = {
    "black-and-white": {"buttercup-festival", "chainsawsuit", "kelly", "sarahs-scribbles", "xkcd"},
    "comic": {
        "barsotti",
        "blitt",
        "buttercup-festival",
        "chainsawsuit",
        "kelly",
        "louder-and-smarter",
        "perry-bible-fellowship",
        "rose-mosco",
        "roz-chast",
        "sarahs-scribbles",
        "tom-gauld",
        "xkcd",
    },
    "new-yorker": {"barsotti", "blitt", "roz-chast"},
    "image": {"comic"},
    "post": {"twitter", "tumblr"},
    "text": {"dril"},
    "twitter": {"dril", "naomi-wolf"},
}

for key, values in _implied_categories.items():
    _known_categories.add(key)
    _known_categories.update(values)

_exclusive_categories = [
    {"color", "black-and-white"},
    {"mostly-black", "mostly-white"},
    {"image", "text"},
    {
        "buttercup-festival",
        "chainsawsuit",
        "kelly",
        "kreuger",
        "louder-and-smarter",
        "perry-bible-fellowship",
        "rose-mosco",
        "sarahs-scribbles",
    },
    {"dril", "naomi-wolf"},
    {"twitter", "tumblr"},
]

for categories in _exclusive_categories:
    _known_categories.update(categories)


class ImageData:
    """Wrapper for metadata about the different image files.

    "images" default to being "color" but can be set to "black-and-white"
    "posts" default to being "text"

    """

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

        unknown_categories = self.categories - _known_categories
        if len(unknown_categories) > 0:
            raise ValueError(unknown_categories)

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


@dataclass(frozen=True)
class Link:
    url: str
    text: str

    @property
    def n_lines(self):
        return self.text.count("<br>") + 1

    def table_cell(self) -> str:
        return f'<a href="{self.url}">{self.text}</a>'


with open("links.csv") as infile:
    links = [Link(row["url"], row["text"]) for row in csv.DictReader(infile)]
total_lines = sum(link.n_lines for link in links)


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
    remaining_images_in_order: list[str], row_length: int, image_filters: list[Callable[[str, str], bool]]
) -> list[str]:
    """Extract the next row of images from the sorted list of remaining images.

    Parameters
    ----------
    remaining_images_in_order : list[str]
        The remaining images to be collected into table rows, highest priority images first.

    row_length : int
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

    while len(row) < row_length and len(remaining_images_in_order) > 0:
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
    shortest_to_longest: list[str], row_length: int, max_pixel_diff: int
) -> list[list[str | None]]:
    """Generate the interior portions of a table of image names.

    Parameters
    ----------
    shortest_to_longest: list[str]
        Sorted list of image names
    row_length : int
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
                    row_length,
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
                    row_length,
                    [
                        lambda last_key, key: height_difference(last_key, key) > max_pixel_diff,
                        lambda last_key, key: height_difference(last_key, key) >= -max_pixel_diff,
                    ],
                )
            )
            shortest_to_longest = list(reversed(longest_to_shortest))

        counter += 1

    return rows


def generate_link_lines(row_length: int) -> list[list[str]]:
    """Generate a table of links."""
    n_lines_per_row = int(total_lines // row_length)
    row_blocks = [[]]
    lines_in_chunk = 0
    for link in links:
        lines_in_chunk += link.n_lines
        if lines_in_chunk > n_lines_per_row:
            lines_in_chunk = 0
            row_blocks.append([])
        row_blocks[-1].append(link.table_cell())
    return row_blocks


def generate_table(row_length: int, max_height_difference: int) -> list[list[str | None]]:
    """Generate a meta-table object

    The first row is a list of links and each cell in the other rows is an image.

    """
    first_row = [[]]
    n_lines_per_row = int(total_lines // row_length)
    lines_in_chunk = 0
    for link in links:
        lines_in_chunk += link.n_lines
        if lines_in_chunk > n_lines_per_row:
            lines_in_chunk = 0
            first_row.append([])
        first_row[-1].append(link)

    shortest_to_longest = sorted(
        [key for key in image_dict.keys() if key != "butts"], key=lambda x: image_dict[x].thumbnail.height
    )

    rows = [first_row] + generate_table_interior(shortest_to_longest, row_length, max_height_difference)

    if len(rows[-1]) == row_length:
        rows.append([None] * (row_length - 1) + ["butts"])
    else:
        rows[-1] = rows[-1] + [None] * (row_length - 1 - len(rows[-1])) + ["butts"]

    return rows


def table_to_str(rows: list[list[list[Link] | str | None]]) -> str:
    """Convert a table of rows into an HTML table string."""
    html_strings = [8 * " " + "<table>"]
    for row in rows:
        html_strings.append(12 * " " + "<tr>")
        for cell in row:
            if cell is None:
                html_strings.append(16 * " " + "<td></td>")
            elif isinstance(cell, str):
                html_strings.append(16 * " " + image_dict[cell].table_cell())
            else:
                html_strings.append(16 * " " + "<td>")
                html_strings.append(20 * " " + cell[0].table_cell())
                for link in cell[1:]:
                    html_strings.append(20 * " " + "<br>" + link.table_cell())
                html_strings.append(16 * " " + "</td>")
        html_strings.append(12 * " " + "</tr>")
    html_strings.append(8 * " " + "</table>")

    return "\n".join(html_strings)


def main() -> None:
    """Entry point."""

    parser = argparse.ArgumentParser(description="Generate a chunk of an HTML table.")
    parser.add_argument("--row_length", help="Number of objects per row", type=int, default=4)
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

    print(table_to_str(generate_table(args.row_length, args.max_height_difference)))


if __name__ == "__main__":
    main()
