# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TODO(triviaqa): Add a description here."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
from absl import logging
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
# TODO(triviaqa): BibTeX citation
_CITATION = """
@article{2017arXivtriviaqa,
       author = {{Joshi}, Mandar and {Choi}, Eunsol and {Weld},
                 Daniel and {Zettlemoyer}, Luke},
        title = "{triviaqa: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension}",
      journal = {arXiv e-prints},
         year = 2017,
          eid = {arXiv:1705.03551},
        pages = {arXiv:1705.03551},
archivePrefix = {arXiv},
       eprint = {1705.03551},
}
"""
_DOWNLOAD_URL = (
    "http://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz")
_TOP_LEVEL_DIR = "triviaqa-rc/qa"
_TRAIN_FILE_FORMAT = os.path.join(_TOP_LEVEL_DIR, "*-train.json")
_DEV_FILE_FORMAT = os.path.join(_TOP_LEVEL_DIR, "*-dev.json")
_HELDOUT_FILE_FORMAT = os.path.join(_TOP_LEVEL_DIR,
                                    "*-test-*without-answers.json")

_DESCRIPTION = """
triviaqa is a reading comprehension dataset containing over 650K
question-answer-evidence triples. triviaqa includes 95K question-answer
pairs authored by trivia enthusiasts and independently gathered evidence
documents, six per question on average, that provide high quality distant
supervision for answering the questions.
"""


def _train_data_filenames(tmp_dir):
  return tf.io.gfile.glob(os.path.join(tmp_dir, _TRAIN_FILE_FORMAT))


def _dev_data_filenames(tmp_dir):
  return tf.io.gfile.glob(os.path.join(tmp_dir, _DEV_FILE_FORMAT))


def _test_data_filenames(tmp_dir):
  return tf.io.gfile.glob(os.path.join(tmp_dir, _HELDOUT_FILE_FORMAT))


class Triviaqa(tfds.core.GeneratorBasedBuilder):
  """TODO(triviaqa): Short description of my dataset."""

  # TODO(triviaqa): Set up version.
  VERSION = tfds.core.Version("0.1.0")

  def _info(self):
    # TODO(triviaqa): Specifies the tfds.core.DatasetInfo object
    #    search_results = None
    #    if is_web_data:
    #      search_results = tfds.features.Sequence({
    #          "Description": tfds.features.Text(),
    #          "Filename": tfds.features.Text(),
    #          "Rank": tf.int32,
    #          "Title": tfds.features.Text(),
    #          "Url": tfds.features.Test(),
    #      })
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            "question":
                tf.string,
            "question_id":
                tfds.features.Text(),
            "question_source":
                tfds.features.Text(),
            "entity_pages":
                tfds.features.Sequence({
                    "doc_source": tfds.features.Text(),
                    "file_name": tfds.features.Text(),
                    "title": tfds.features.Text(),
                }),
            "answer":
                tfds.features.FeaturesDict({
                    "aliases":
                        tfds.features.Sequence(tfds.features.Text()),
                    "matched_wiki_entity_name":
                        tfds.features.Text(),
                    "normalized_aliases":
                        tfds.features.Sequence(tfds.features.Text()),
                    "normalized_matched_wiki_entity_name":
                        tfds.features.Text(),
                    "normalized_value":
                        tfds.features.Text(),
                    "type":
                        tfds.features.Text(),
                    "value":
                        tfds.features.Text(),
                }),
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=None,
        # Homepage of the dataset for documentation
        urls=["http://nlp.cs.washington.edu/triviaqa/"],
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    # TODO(triviaqa): Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    trivia_path = dl_manager.download_and_extract(_DOWNLOAD_URL)

    train_files = _train_data_filenames(trivia_path)
    dev_files = _dev_data_filenames(trivia_path)
    test_files = _test_data_filenames(trivia_path)
    # Generate vocabulary from training data if SubwordTextEncoder configured
    self.info.features["text"].maybe_build_from_corpus(
        self._vocab_text_gen(train_files))

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            num_shards=100,
            gen_kwargs={"files": train_files}),
        tfds.core.SplitGenerator(
            name=tfds.Split.DEV,
            num_shards=50,
            gen_kwargs={"files": dev_files}),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            num_shards=50,
            gen_kwargs={"files": test_files}),
    ]

  def _generate_examples(self, filepath):
    """This function returns the examples in the raw (text) form."""
    logging.info("generating examples from = %s", filepath)
    with tf.io.gfile.GFile(filepath) as f:
      triviaqa = json.load(f)
      for article in triviaqa["Data"]:
        answer = article["Answer"]
        aliases = [alias for alias in answer["Aliases"].strip()]
        matched_wiki_entity_name = answer["MatchedWikiEntityName"].strip()
        normalized_aliases = [
            alias for alias in answer["NormalizedAliases"].strip()
        ]
        normalized_matched_wiki_entity_name = answer[
            "NormalizedMatchedWikiEntityName"].strip()
        normalized_value = answer["NormalizedValue"].strip()
        type_ = answer["Type"].strip()
        value = answer["Value"].strip()
        question = article["Question"].strip()
        question_id = article["QuestionId"]
        question_source = article["QuestionSource"].strip()

        doc_sources = [
            entitypage["Docsource"] for entitypage in article["EntityPages"]
        ]
        file_names = [
            entitypage["Filename"] for entitypage in article["EntityPages"]
        ]
        titles = [entitypage["Title"] for entitypage in article["EntityPages"]]

        yield {
            "entity_pages": {
                "doc_source": doc_sources,
                "file_name": file_names,
                "title": titles,
            },
            "question": question,
            "questionid": question_id,
            "question_source": question_source,
            "answer": {
                "aliases":
                    aliases,
                "matched_wiki_entity_name":
                    matched_wiki_entity_name,
                "normalized_aliases":
                    normalized_aliases,
                "normalized_matched_wiki_entity_name":
                    normalized_matched_wiki_entity_name,
                "normalized_value":
                    normalized_value,
                "type":
                    type_,
                "value":
                    value,
            },
        }
