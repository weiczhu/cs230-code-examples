import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")


class Node:
    def __init__(self, label=None, data=None):
        self.label = label
        self.data = data
        self.children = dict()

    def addChild(self, key, data=None):
        if not isinstance(key, Node):
            self.children[key] = Node(key, data)
        else:
            self.children[key.label] = key

    def __getitem__(self, key):
        return self.children[key]


class Trie:
    def __init__(self):
        self.head = Node()
        self.dictionary = dict()

    def __getitem__(self, key):
        return self.head.children[key]

    def add(self, word, type):
        if word == '' or type == '':
            return
        if word == None or type == None:
            raise ValueError('Trie.add requires a not-Null string')

        types = self.dictionary.get(word, [])
        if type not in types:
            types.append(type)
        self.dictionary[word] = types

        current_node = self.head
        word_finished = True

        for i in range(len(word)):
            if word[i] in current_node.children:
                current_node = current_node.children[word[i]]
            else:
                word_finished = False
                break

        # For ever new letter, create a new child node
        if not word_finished:
            while i < len(word):
                current_node.addChild(word[i])
                current_node = current_node.children[word[i]]
                i += 1

        # Let's store the full word at the end node so we don't need to
        # travel back up the tree to reconstruct the word
        current_node.data = word

    def has_word(self, word):
        if word == '':
            return False
        if word == None:
            raise ValueError('Trie.has_word requires a not-Null string')

        # Start at the top
        current_node = self.head
        exists = True
        for letter in word:
            if letter in current_node.children:
                current_node = current_node.children[letter]
            else:
                exists = False
                break

        # Still need to check if we just reached (part of) a word like 't'
        # that isn't actually a full word in our dictionary
        if exists:
            if current_node.data == None:
                exists = False

        return exists

    def start_with_prefix(self, prefix):
        """ Returns a list of all words in tree that start with prefix """
        words = list()
        if prefix == None:
            raise ValueError('Requires not-Null prefix')

        # Determine end-of-prefix node
        top_node = self.head
        for letter in prefix:
            if letter in top_node.children:
                top_node = top_node.children[letter]
            else:
                # Prefix not in tree, go no further
                return words

        # Get words under prefix
        if top_node == self.head:
            queue = [node for key, node in top_node.children.items()]
        else:
            queue = [top_node]

        # Perform a breadth first search under the prefix
        # A cool effect of using BFS as opposed to DFS is that BFS will return
        # a list of words ordered by increasing length
        while queue:
            current_node = queue.pop()
            if current_node.data != None:
                # Isn't it nice to not have to go back up the tree?
                words.append(current_node.data)

            # BFS
            queue = [node for key, node in current_node.children.items()] + queue

            # DFS
            # queue = queue + [node for key, node in current_node.children.items()]

        return words

    def get_data(self, word):
        """ This returns the 'data' of the node identified by the given word """
        if not self.has_word(word):
            raise ValueError('{} not found in trie'.format(word))

        # Race to the bottom, get data
        current_node = self.head
        for letter in word:
            current_node = current_node[letter]

        types = self.dictionary[current_node.data]
        return (current_node.data, types)


def pos_tag(sentence):
    segments = sentence.split(' ')

    annotate_sents = []
    for i in range(len(segments)):
        for j in range(i, len(segments)):
            words_str = ' '.join(segments[i:j + 1])
            try:
                print(trie.get_data(words_str))
                annotate_sents.append(trie.get_data(words_str))
            except ValueError as e:
                # print("An exception was raised, skipping a word: {}".format(e))
                pass

    return annotate_sents


def pred_suggest(sentence):
    to_sugguest = sentence.split(' ')[-1]

    print('word suggestion for prefix \'{}\':'.format(sentence))
    print(trie.start_with_prefix(to_sugguest))
    print(len(trie.start_with_prefix('')))

    suggest_words = trie.start_with_prefix('')

    return suggest_words


if __name__ == '__main__':
    global trie
    trie = Trie()

    # Load the dictionary from the dataset into params
    args = parser.parse_args()
    dictionary_path = os.path.join(args.data_dir, 'dictionary.txt')
    assert os.path.isfile(dictionary_path), "No words file found at {}, run build.py".format(dictionary_path)

    with open(dictionary_path) as f:
        for line in f:
            word, type = line.strip().split('::')
            trie.add(word, type)

    # if trie.has_word('tamagawa denenchofu'):
    #     print("'tamagawa denenchofu' in trie: ", trie.get_data('tamagawa denenchofu'))
    #
    # print('word suggestion for prefix \'ta\':')
    # print(trie.start_with_prefix('ta'))

    sentence = 'shall we go to tamagawa denenchofu setagaya-ku tokyo 158-0085 for lunch ?'
    annotate_sents = pos_tag(sentence)

    sugguest_test = sentence[:18]
    suggest_words = pred_suggest(sugguest_test)
