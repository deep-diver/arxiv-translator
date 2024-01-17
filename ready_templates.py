import sys
import os


PAPER_DIR = "papers"
HTML_TEMPLATE_FN = "assets/html_template.html"

GITHUB_ID="kh-kim"
REPO_NAME="arxiv-translator"

def main(en_mmd_fn, html_tpl_fn, paper_dir):
    en_mmd_lines = []
    with open(en_mmd_fn, "r") as f:
        for line in f:
            en_mmd_lines.append(line.replace("\n", ""))

    ko_mmd_fn = en_mmd_fn.replace(".mmd", ".ko.mmd")
    ko_mmd_lines = []
    with open(ko_mmd_fn, "r") as f:
        for line in f:
            ko_mmd_lines.append(line.replace("\n", ""))

    arxiv_id = en_mmd_fn.split("/")[-1].split("_")[0]

    html_lines = []
    with open(html_tpl_fn, "r") as f:
        for line in f:
            html_lines.append(line.replace("\n", ""))

    def wrap(line):
        return "      '" + line.replace("\\", "\\\\").replace("'", "\\'") + "\\n' +"

    en_result_html_lines = html_lines[:6] + [wrap(line) for line in en_mmd_lines] + html_lines[6:]
    en_result_html = "\n".join(en_result_html_lines)

    with open(os.path.join(paper_dir, arxiv_id, "paper.en.html"), "w") as f:
        f.write(en_result_html)

    ko_result_html_lines = html_lines[:6] + [wrap(line) for line in ko_mmd_lines] + html_lines[6:]
    ko_result_html = "\n".join(ko_result_html_lines)

    with open(os.path.join(paper_dir, arxiv_id, "paper.ko.html"), "w") as f:
        f.write(ko_result_html)

if __name__ == "__main__":
    en_mmd_fn = sys.argv[1]
    html_tpl_fn = sys.argv[2]
    paper_dir = sys.argv[3]

    main(en_mmd_fn, html_tpl_fn, paper_dir)
