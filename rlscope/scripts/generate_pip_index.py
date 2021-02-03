"""
Generate an html file that contains all the wheel file releases on github.
"""
# TODO: run this script as part of a post-commit hook for master branch / tag-ing
import argparse
import textwrap
import sys
import pprint
import csv
import html
import urllib
import os
import re
from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

from rlscope.profiler.rlscope_logging import logger
from rlscope import py_config
from rlscope.profiler.util import pprint_msg

# pip install PyGithub
from github import Github

class RLScopeRelease:
    def __init__(self, tag, cuda_version=None, py_version='3', platform='manylinux1_x86_64',
                 wheel_file=None):
        self.tag = tag
        if wheel_file is None:
            self.cuda_version = cuda_version
            self.py_version = py_version
            self.platform = platform

            self.wheel_file = self._wheel_file(self.tag, self.cuda_version, self.py_version, self.platform)
        else:
            self.wheel_file = wheel_file

    @staticmethod
    def from_github_release(release):
        tag = release.tag_name
        rlscope_releases = []
        for asset in release.raw_data['assets']:
            wheel_file = urllib.parse.unquote(_b(asset['browser_download_url']))
            if not re.search(r'\.whl$', wheel_file):
                continue
            rlscope_release = RLScopeRelease(tag=tag, wheel_file=wheel_file)
            rlscope_releases.append(rlscope_release)
        return rlscope_releases

    def _wheel_file(self, tag, cuda_version, py_version, platform):
        version = re.sub(r'^v', '', tag)
        cuda = "cu" + re.sub(r'\.', '', cuda_version)
        return "rlscope-{version}+{cuda}-py{pyver}-none-{platform}.whl".format(
            version=version,
            cuda=cuda,
            pyver=py_version,
            platform=platform,
        )

    @property
    def wheel_url(self):
        """
        e.g.
        https://github.com/UofT-EcoSystem/rlscope/releases/download/v0.0.1/rlscope-0.0.1+cu101-py3-none-manylinux1_x86_64.whl
        """
        wheel_url = "/".join([
            py_config.GIT_REPO_URL,
            "releases/download/{tag}".format(
                tag=self.tag,
            ),
            self.wheel_file,
        ])
        return wheel_url

    def __repr__(self):
        return "{klass}(url={url})".format(
            klass=self.__class__.__name__,
            url=self.wheel_url,
        )

# TODO: scrape github repo releases page to automatically generate this.
# RLSCOPE_RELEASES = [
#     RLScopeRelease('v0.0.1', '10.1'),
#     RLScopeRelease('v0.0.1', '11.0'),
# ]

class PipIndexGenerator:
    def __init__(self, args):
        self.args = args

    def generate_html(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.rlscope_releases = self.scrape_releases()
        with open(path, 'w') as f:
            for release in self.rlscope_releases:
                f.write(textwrap.dedent("""\
                <a href="{url}">{wheel}</a><br/>
                """.format(
                    url=html.escape(release.wheel_url),
                    wheel=release.wheel_file,
                )))
        logger.info("Output html file @ {path}".format(path=path))

    def scrape_releases(self):
        """
        Scrap https://github.com/UofT-EcoSystem/rlscope for uploaded wheel files.
        :return:
        """

        g = Github()
        repo = g.get_repo("UofT-EcoSystem/rlscope")

        releases = repo.get_releases()
        rlscope_releases = []
        for release in releases:
            rlscope_releases.extend(RLScopeRelease.from_github_release(release))

        logger.info("Found releases at {url}: {msg}".format(
            url=py_config.GIT_REPO_URL,
            msg=pprint_msg(rlscope_releases)))

        return rlscope_releases


def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__.lstrip().rstrip()),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--output",
                        # required=True,
                        default=_j(py_config.ROOT, 'whl', 'index.html'),
                        help=textwrap.dedent("""\
                        HTML file to output
                        """))
    parser.add_argument("--debug",
                        action='store_true',
                        help=textwrap.dedent("""\
                        Debug
                        """))
    args = parser.parse_args()

    generator = PipIndexGenerator(args)
    generator.generate_html(args.output)

if __name__ == '__main__':
    main()
