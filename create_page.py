"""A li'l script for generating a sorted table of images."""

import argparse
import csv
from dataclasses import dataclass
from functools import cache, cached_property
import math
from pathlib import Path
import subprocess
from typing import Callable, Iterator, TypeVar

from PIL import Image

RELATIVE_THUMB = Path("images/thumbnail")
RELATIVE_FULL = Path("images/full")

_known_categories = {
    "art",
    "babadook",
    "lubchansky",
    "new-yorker",
}

_implied_categories = {
    "black-and-white": {"buttercup-festival", "chainsawsuit", "kelly", "sarahs-scribbles", "xkcd"},
    "color": {"penny-arcade", "rose-mosco", "webcomic-dot-name"},
    "comic": {
        "barsotti",
        "blitt",
        "buttercup-festival",
        "cat-and-girl",
        "chainsawsuit",
        "kelly",
        "louder-and-smarter",
        "penny-arcade",
        "perry-bible-fellowship",
        "rose-mosco",
        "roz-chast",
        "sarahs-scribbles",
        "tom-gauld",
        "webcomic-dot-name",
        "xkcd",
    },
    "mostly-white": {"chainsawsuit", "sarahs-scribbles", "webcomic-dot-name", "xkcd"},
    "new-yorker": {"barsotti", "blitt", "roz-chast"},
    "image": {"comic"},
    "post": {"twitter", "tumblr"},
    "text": {"dril", "poem"},
    "poem": {"hughes", "nael"},
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

for _exclusive_category in _exclusive_categories:
    _known_categories.update(_exclusive_category)


ImageDataType = TypeVar("ImageDataType")


class ImageData:
    """Wrapper for metadata about the different image files.

    "images" default to being "color" but can be set to "black-and-white"
    "posts" default to being "text"

    """

    def __init__(self, thumbnail_path: Path, full_path: Path, alt_text: str, categories: set[str], fix_missing_thumbnail: bool = True):
        self._thumbnail_path = thumbnail_path
        self._full_path = full_path
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

        if not self.full_path.is_file():
            raise FileNotFoundError(self.full_path)

        if not self.thumbnail_path.is_file() and fix_missing_thumbnail:
            subprocess.run(["magick", str(self.full_path), "-resize", "200x", str(self.thumbnail_path)])

        if not self.thumbnail_path.is_file():
            raise FileNotFoundError(self.thumbnail_path)

        for path in self.full_path, self.thumbnail_path:
            if not path.is_file():
                raise FileNotFoundError(path)

    def __repr__(self) -> str:
        """Return a string representation of the object sufficient to recreate it."""
        return f"ImageData({self.thumbnail_path!r}, {self.full_path!r}, {self.alt_text!r}, {self.categories!r})"

    @property
    def name(self):
        """Return the name of the image."""
        return self.thumbnail_path.stem

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
    def thumbnail(self) -> Image:
        return Image.open(self.thumbnail_path)

    @property
    def alt_text(self) -> str:
        return self._alt_text

    @property
    def categories(self) -> set[str]:
        return self._categories

    @property
    def full_path(self) -> Path:
        """Return the path to the thumbnail image."""
        return self._full_path

    @property
    def thumbnail_path(self) -> Path:
        """Return the path to the thumbnail image."""
        return self._thumbnail_path

    def table_cell(self) -> str:
        return f'<td><a href="{self.full_path}"><img src="{self.thumbnail_path}" alt="{self.alt_text}"></a></td>'

    def __sub__(self, other: ImageDataType) -> int:
        """Return the difference in height between this image and another image."""
        if not isinstance(other, ImageData):
            return NotImplemented
        return self.thumbnail.height - other.thumbnail.height

    def jaccard_similarity(self, other: ImageDataType) -> float:
        if not isinstance(other, ImageData):
            return NotImplemented
        return jaccard_similarity(self, other)


@dataclass(frozen=True)
class Link:
    url: str
    text: str

    @property
    def n_lines(self):
        return self.text.count("<br>") + 1

    def table_cell(self) -> str:
        return f'<a href="{self.url}">{self.text}</a>'


@cache
def jaccard_similarity(*args: ImageData) -> float:
    """Calculate the Jaccard similarity of images based on metadata categories.

    1 indicates two images have identical categories.
    0 indicates two images have no common categories.

    If more than two items are provided, the pairwise similarity of all items will be returned.

    """
    if len(args) < 2:
        raise ValueError(len(args))

    if len(args) > 2:
        return sum(jaccard_similarity(arg_one, arg_two) for arg_one, arg_two in zip(args[:-1], args[1:]))

    arg_one, arg_two = args
    if arg_one.name > arg_two.name:
        return jaccard_similarity(arg_two, arg_one)
    return len(arg_one.categories & arg_two.categories) / len(arg_one.categories | arg_two.categories)


def extracted_row_of_images(
    remaining_images_in_order: list[ImageData],
    row_length: int,
    image_filters: list[Callable[[ImageData, ImageData], bool]],
) -> list[ImageData]:
    """Extract the next row of images from the sorted list of remaining images.

    Parameters
    ----------
    remaining_images_in_order : list[str]
        The remaining images to be collected into table rows, highest priority images first.

    row_length : int
        The number of images to place in the new row.

    image_filters : list[Callable[[ImageData, ImageData], bool]]
        List of image tests. All images passing the first filter should be evaluated before images passing the
        second filter, etc.

    Returns
    -------
    list[ImageData]
        Row of images

    """
    row = [remaining_images_in_order.pop(0)]

    while len(row) < row_length and len(remaining_images_in_order) > 0:
        next_image = remaining_images_in_order[0]

        for image_filter in image_filters:
            candidate_images = []
            for remaining_image_in_order in remaining_images_in_order:
                if image_filter(row[-1], remaining_image_in_order):
                    candidate_images.append(remaining_image_in_order)
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
    shortest_to_longest: list[ImageData], row_length: int, max_pixel_diff: int
) -> list[list[ImageData | None]]:
    """Generate the interior portions of a table of image names.

    Parameters
    ----------
    shortest_to_longest: list[ImageData]
        Sorted list of images
    row_length : int
        Number of images to put in each row of the table
    max_pixel_diff: int
        Maximum height differences for sorting

    Returns
    -------
    list[list[ImageData | None]]
    """
    image_filters = {
        "increasing": [
            lambda prev_image, cur_image: cur_image - prev_image < -max_pixel_diff,
            lambda prev_image, cur_image: cur_image - prev_image <= max_pixel_diff,
        ],
        "decreasing": [
            lambda prev_image, cur_image: cur_image - prev_image > max_pixel_diff,
            lambda prev_image, cur_image: cur_image - prev_image >= -max_pixel_diff,
        ],
    }

    rows = []
    while len(shortest_to_longest) > 0:
        if len(rows) % 2 == 0:
            # Add row that is roughly increasing in height, growing from shortest
            rows.append(extracted_row_of_images(shortest_to_longest, row_length, image_filters["increasing"]))
        else:
            # Add row that is roughly decreasing in height, shrinking from longest
            longest_to_shortest = list(reversed(shortest_to_longest))
            rows.append(extracted_row_of_images(longest_to_shortest, row_length, image_filters["decreasing"]))
            shortest_to_longest = list(reversed(longest_to_shortest))

    return rows


def link_cells(links: list[Link], row_length: int) -> Iterator[tuple[Link, ...]]:
    """Given a row length, yield list of links chunked by total number of lines.

    A link row consists of cells with a certain number of lines. Our goal is a roughly even
    number of lines in each cell across the row.

    """
    if len(links) < 1:
        return tuple()

    remaining_lines = sum(link.n_lines for link in links)
    remaining_cells = row_length
    n_lines_in_current_cell = math.ceil(remaining_lines / remaining_cells)
    links_iter = iter(links)
    cell = [next(links_iter)]
    lines_in_cell = cell[0].n_lines
    for link in links_iter:
        if lines_in_cell + link.n_lines > n_lines_in_current_cell:
            yield tuple(cell)
            remaining_lines -= lines_in_cell
            remaining_cells -= 1
            n_lines_in_current_cell = math.ceil(remaining_lines / max(remaining_cells, 1))
            cell = [link]
            lines_in_cell = cell[-1].n_lines
        else:
            cell.append(link)
            lines_in_cell += cell[-1].n_lines
    if len(cell) > 0:
        yield tuple(cell)


def table_header_row(header: str) -> str:
    return 12 * " " + f"<tr><td><strong>{header}</strong></td></tr>"


def generate_page(
    links_dict: dict[str, list[Link]], images: list[ImageData], row_length: int, max_height_difference: int
) -> str:
    """Generate the web page."""

    html_strings = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        4 * " " + '<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible" rel="stylesheet">',
        4 * " " + '<link href="style.css" rel="stylesheet" type="text/css">',
        4 * " " + "<title>A table relaxes the eyes</title>",
        "</head>",
        "<body>",
        4 * " " + '<div class="text">',
        8 * " " + "<table>",
    ]
    for key, links in links_dict.items():
        html_strings.append(12 * " " + "<tr>")
        html_strings.append(
            16 * " " + f"<td><strong>{' '.join([word.capitalize() for word in key.split('_')])}</strong></td>"
        )
        for cells in link_cells(links, row_length - 1):
            html_strings.append(16 * " " + "<td>")
            html_strings.append(20 * " " + cells[0].table_cell())
            for cell in cells[1:]:
                html_strings.append(20 * " " + "<br>" + cell.table_cell())
            html_strings.append(16 * " " + "</td>")
        html_strings.append(12 * " " + "</tr>")

    butts = next(image for image in images if image.name == "butts")
    shortest_to_longest = sorted(
        [image for image in images if image.name != "butts"], key=lambda image: image.thumbnail.height
    )

    rows = generate_table_interior(shortest_to_longest, row_length, max_height_difference)
    if len(rows[-1]) == row_length:
        rows.append([None] * (row_length - 1) + [butts])
    else:
        rows[-1] = rows[-1] + [None] * (row_length - 1 - len(rows[-1])) + [butts]

    for row in rows:
        html_strings.append(12 * " " + "<tr>")
        html_strings += [16 * " " + ("<td></td>" if image is None else image.table_cell()) for image in row]
        html_strings.append(12 * " " + "</tr>")

    html_strings.append(8 * " " + "</table>")
    html_strings.append(4 * " " + "</div>")
    html_strings.append("</body>")
    html_strings.append("</html>")

    return "\n".join(html_strings)


def load_images(image_directory: str | Path) -> list[ImageData]:
    """Check for images, ingest metadata, and return as a dict."""

    image_directory = Path(image_directory)
    metadata_path = (image_directory / "metadata.csv").resolve(strict=True)
    return [
        ImageData(
            image_directory / "thumbnail" / f"{row['name']}.png",
            image_directory / "full" / f"{row['name']}.png",
            row["alt_text"],
            {category.strip() for category in row["categories"].split(",")},
        )
        for row in csv.DictReader(metadata_path.open("r"))
    ]


def load_links_dict(links_directory: str | Path) -> dict[str, list[Link]]:
    """Ingest links files into a dict and return it."""
    all_links = {}
    for path in sorted(Path(links_directory).resolve(strict=True).glob("*.csv")):
        all_links[path.name[: -len(".csv")]] = [Link(row["url"], row["text"]) for row in csv.DictReader(path.open("r"))]
    return all_links


def main() -> None:
    """Entry point."""

    parser = argparse.ArgumentParser(description="Generate a chunk of an HTML table.")
    parser.add_argument("--links_directory", type=Path, default=Path("links"))
    parser.add_argument("--image_directory", type=Path, default=Path("images"))
    parser.add_argument("--row_length", help="Number of objects per row", type=int, default=5)
    parser.add_argument(
        "--max_height_difference", help="Maximum difference in height between images", type=int, default=5
    )
    args = parser.parse_args()

    if args.row_length < 2:
        raise ValueError(args.row_length)

    links_dict = load_links_dict(args.links_directory)
    images = load_images(args.image_directory)

    with open("index.html", "w") as outfile:
        outfile.write(generate_page(links_dict, images, args.row_length, args.max_height_difference))
        outfile.write("\n")


if __name__ == "__main__":
    main()
