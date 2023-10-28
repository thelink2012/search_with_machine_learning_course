import argparse
import multiprocessing
import glob
from tqdm import tqdm
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')

ps = PorterStemmer()

def transform_name(product_name):
    product_name = product_name.lower()
    tokens = word_tokenize(product_name)
    #tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)

# Directory for product data
directory = r'/workspace/datasets/product_data/products/'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")
general.add_argument("--label", default="id", help="id is default and needed for downsteam use, but name is helpful for debugging")

general.add_argument("--min_products", default=0, type=int, help="The minimum number of products per category (default is 0).")
general.add_argument("--promote", action=argparse.BooleanOptionalAction, default=False, help="Promote categories to its parent instead of discarding (needs --min_products set).")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input
min_products = args.min_products
names_as_labels = False
if args.label == 'name':
    names_as_labels = True

def filter_min_products(all_labels):
    counter = dict()
    for cat, name in all_labels:
        counter[cat] = 0. # init cat key
    for cat, name in all_labels:
        counter[cat] += 1
    return [(cat, name) for cat, name in all_labels if counter[cat] >= min_products]


def _exists_and_needs_promotion(cat, counter):
    c = counter.get(cat, 0)
    return _exists_and_needs_promotion_v(c)

def _exists_and_needs_promotion_v(count):
    return count > 0 and count < min_products

def _read_category_info():
    tree = ET.parse('/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml')
    root = tree.getroot()
    parent_of_cat = dict()
    indepth_children_of_cat = dict()
    for child in root:
        cat_id = child.find('id').text

        cat_path = [cpath.find('id').text for cpath in child.find('path')]
        cat_path = [c for c in cat_path if c != cat_id]
        cat_parent = cat_path[-1] if len(cat_path) > 0 else None
        parent_of_cat[cat_id] = cat_parent

        cat_subs = set(subcat.find('id').text for subcat in child.find('subCategories'))
        indepth_children_of_cat[cat_id] = cat_subs
    return parent_of_cat, indepth_children_of_cat


def promote_categories(all_labels):
    print("promoting categories")

    counter = dict()
    for cat, name in all_labels:
        counter[cat] = 0. # init cat key
    for cat, name in all_labels:
        counter[cat] += 1

    parent_of_cat, indepth_children_of_cat =_read_category_info()

    while any(_exists_and_needs_promotion_v(v) for v in counter.values()):
        print(f"still exists {sum(v < min_products for v in counter.values())} categories that need promotion")
        for i in range(len(all_labels)):
            if all_labels[i] is None:  # excluded
                continue
            cat, name = all_labels[i]
            if counter[cat] >= min_products:
                continue  # do not promote if does not need promotion
            if cat not in indepth_children_of_cat:
                 # missing data in categories XML, cannot do much about it
                assert cat not in parent_of_cat
                counter[cat] -= 1
                all_labels[i] = None
                continue
            if any(_exists_and_needs_promotion(ichild, counter) for ichild in indepth_children_of_cat[cat]):
                continue  # do not promote if there exists children categories that still need promotion
            new_cat = parent_of_cat[cat]
            if new_cat is None:  # no parent
                counter[cat] -= 1
                all_labels[i] = None # cannot promote anymore, exclude
            else:
                counter[cat] -= 1
                if new_cat not in counter:
                    counter[new_cat] = 0
                counter[new_cat] += 1
                all_labels[i] = (new_cat, name)
    
    all_labels = [x for x in all_labels if x is not None]
    all_labels = [(cat, name) for cat, name in all_labels if cat != 'cat00000']
    return all_labels




def _label_filename(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    labels = []
    for child in root:
        # Check to make sure category name is valid and not in music or movies
        if (child.find('name') is not None and child.find('name').text is not None and
            child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
            child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None and
            child.find('categoryPath')[0][0].text == 'cat00000' and
            child.find('categoryPath')[1][0].text != 'abcat0600000'):
              # Choose last element in categoryPath as the leaf categoryId or name
              if names_as_labels:
                  cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][1].text.replace(' ', '_')
              else:
                  cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text
              # Replace newline chars with spaces so fastText doesn't complain
              name = child.find('name').text.replace('\n', ' ')
              labels.append((cat, transform_name(name)))
    return labels

if __name__ == '__main__':
    files = glob.glob(f'{directory}/*.xml')
    print("Writing results to %s" % output_file)
    with multiprocessing.Pool() as p:
        all_labels_partitioned = tqdm(p.imap(_label_filename, files), total=len(files))
        all_labels = []
        for label_list in all_labels_partitioned:
            all_labels.extend(label_list)
        del all_labels_partitioned
    if min_products > 0:
        if args.promote:
            all_labels = promote_categories(all_labels)
        else:
            all_labels = filter_min_products(all_labels)
    with open(output_file, 'w') as output:
        for (cat, name) in all_labels:
            output.write(f'__label__{cat} {name}\n')
