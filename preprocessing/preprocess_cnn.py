'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import os
import hashlib
import csv

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  # acceptable ways to end a sentence

all_train_urls = "../../data/cnndm/url_lists/all_train.txt"
all_val_urls = "../../data/cnndm/url_lists/all_val.txt"
all_test_urls = "../../data/cnndm/url_lists/all_test.txt"

cnn_tokenized_stories_dir = "../../data/cnndm/cnn_stories_tokenized"
finished_files_dir = "../../data/cnndm/cnn/"


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    # print line[-1]
    return line + " ."


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def get_art_abs(story_file):
    lines = read_text_file(story_file)

    # Lowercase everything
    #lines = [line.lower() for line in lines]

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = ' '.join(article_lines)

    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    abstract = ' '.join([sent for sent in highlights])

    return article, abstract


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def stratify_and_write_to_outfile(url_file, finished_files_dir, outfile_name):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
    url_list = read_text_file(url_file)
    url_hashes = get_url_hashes(url_list)

    story_fnames = [s + ".story" for s in url_hashes]
    num_stories = len(story_fnames)

    outfile = open(os.path.join(finished_files_dir, outfile_name), 'w')
    fieldnames = ['id', 'article', 'summary']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, delimiter='\t')
    writer.writeheader()

    domain_wise_data = {}

    for idx, s in enumerate(story_fnames):
        if idx % 1000 == 0:
            print("Writing story %i of %i; %.2f percent done" % (
            idx, num_stories, float(idx) * 100.0 / float(num_stories)))

        # Look in the tokenized story dirs to find the .story file corresponding to this url
        if os.path.isfile(os.path.join(cnn_tokenized_stories_dir, s)):
            story_file = os.path.join(cnn_tokenized_stories_dir, s)

            # Get strings
            article, abstract = get_art_abs(story_file)
            row_output = {'article': article, 'summary': abstract, 'id': s}

            domain = url_list[idx].split('/')[-3]

            if domain not in domain_wise_data.keys():
                domain_wise_data[domain] = []
            domain_wise_data[domain].append(row_output)

            writer.writerow(row_output)

    return
    print(outfile_name)
    for domain in domain_wise_data:
        print(domain)
        print(len(domain_wise_data[domain]))
        if not os.path.exists(os.path.join(finished_files_dir, domain)):
            os.makedirs(os.path.join(finished_files_dir, domain))

        outfile = open(os.path.join(finished_files_dir, domain, outfile_name), 'w')
        fieldnames = ['id', 'article', 'summary']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, delimiter='\t')
        writer.writeheader()

        for row in domain_wise_data[domain]:
            writer.writerow(row)

        outfile.close()


    print("Finished writing file %s\n" % outfile_name)


def write_to_outfile(url_file, out_file):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
    url_list = read_text_file(url_file)
    url_hashes = get_url_hashes(url_list)
    story_fnames = [s+".story" for s in url_hashes]
    num_stories = len(story_fnames)

    outfile = open(out_file, 'w')
    fieldnames = ['id', 'article', 'summary']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, delimiter='\t')
    writer.writeheader()

    for idx, s in enumerate(story_fnames):
        if idx % 1000 == 0:
            print("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))

        # Look in the tokenized story dirs to find the .story file corresponding to this url
        if os.path.isfile(os.path.join(cnn_tokenized_stories_dir, s)):
            story_file = os.path.join(cnn_tokenized_stories_dir, s)
            """
            elif os.path.isfile(os.path.join(dm_tokenized_stories_dir, s)):
                story_file = os.path.join(dm_tokenized_stories_dir, s)"""
        else:
            continue

        # Get the strings to write to .bin file
        article, abstract = get_art_abs(story_file)
        row_output = {'article': article, 'summary': abstract, 'id': s}
        writer.writerow(row_output)

    print("Finished writing file %s\n" % out_file)


if __name__ == '__main__':
    if not os.path.exists(finished_files_dir):
        os.makedirs(finished_files_dir)
    write_to_outfile(all_val_urls, os.path.join(finished_files_dir, 'dev.tsv'))
    write_to_outfile(all_test_urls, os.path.join(finished_files_dir, 'test.tsv'))
    write_to_outfile(all_train_urls, os.path.join(finished_files_dir, 'train.tsv'))
