import os

from wrapper import Sequence
from utils import load_data_and_labels, load_glove


if __name__ == '__main__':
    DATA_ROOT = os.path.join(os.path.dirname(__file__), os.environ["data_dir"])
    EMBEDDING_PATH = os.path.join(os.path.dirname(__file__), os.environ["glove_dir"])

    train_path = os.path.join(DATA_ROOT, 'train.txt')
    valid_path = os.path.join(DATA_ROOT, 'valid.txt')
    test_path = os.path.join(DATA_ROOT, 'test.txt')

    print('Loading data...')
    x_train, y_train = load_data_and_labels(train_path)
    x_valid, y_valid = load_data_and_labels(valid_path)
    x_test, y_test = load_data_and_labels(test_path)
    print(len(x_train), 'train sequences')
    print(len(x_valid), 'valid sequences')
    print(len(x_test), 'test sequences')

    embeddings = load_glove(EMBEDDING_PATH)

    # Use pre-trained word embeddings
    model = Sequence(cell_type=os.environ['cell_type'], embeddings=embeddings, initial_vocab=embeddings.keys())
    # print(model.trainable_weights)

    model.fit(x_train, y_train, x_valid, y_valid, epochs=30)

    print('Testing the model...')
    print(model.score(x_test, y_test))
