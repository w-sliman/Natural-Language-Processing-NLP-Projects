import tensorflow as tf
import numpy as np
from typing import List, Optional, Tuple, Union

class NamedEntityRecognition:
    """
    A class for Named Entity Recognition using Tensorflow LSTM layers.

    Attributes:
    embedding_dim (int): Dimension of word embeddings.
    num_lstm_layers (int): Number of LSTM layers.
    bidirectional_lstms (bool): Whether to use bidirectional LSTMs.
    random_state (Optional[int]): Random seed for reproducibility.
    sentence_vectorizer (Optional[tf.keras.layers.TextVectorization]): Text vectorizer for sentences.
    vocab (Optional[List[str]]): Vocabulary list.
    tag_list (Optional[List[str]]): List of tags.
    tag_map (Optional[dict]): Mapping from tags to indices.
    model (Optional[tf.keras.Model]): The LSTM model.
    """
    def __init__(
        self,
        embedding_dim: int = 50,
        num_lstm_layers: int = 2,
        bidirectional_lstms: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_lstm_layers = num_lstm_layers
        self.bidirectional_lstms = bidirectional_lstms
        self.random_state = random_state

        self.sentence_vectorizer: Optional[tf.keras.layers.TextVectorization] = None
        self.vocab: Optional[List[str]] = None
        self.tag_list: Optional[List[str]] = None
        self.tag_map: Optional[dict] = None
        self.model: Optional[tf.keras.Model] = None

        if random_state:
            self._set_random_seed(self.random_state)

    def _set_random_seed(self, seed: int) -> None:
        """Sets the random seed for reproducibility."""
        tf.random.set_seed(seed)
        np.random.seed(seed)

    def fit(
        self,
        sentences: List[str],
        labels: List[List[str]],
        epochs: int,
        #validation_data: Optional[Tuple[List[str], List[List[str]]]] = None,
        validation_data: Optional[List[Union[List[str], List[List[str]]]]] = None,
        batch_size: int = 64,
    ) -> 'NamedEntityRecognition':
        """
        Trains the NER model on the provided sentences and labels.

        Args:
        sentences (List[str]): List of sentences for training.
        labels (List[List[str]]): List of label sequences corresponding to the sentences.
        epochs (int): Number of training epochs.
        validation_data (Optional[List[Union[List[str], List[List[str]]]]]): Validation data.
        batch_size (int): Size of the training batches.

        Returns:
        NamedEntityRecognition: The trained NER model.
        """
        self.sentence_vectorizer, self.vocab = self._get_sentence_vectorizer(sentences)
        self.tag_list = self._get_tags(labels)
        self.tag_map = self._make_tag_map(self.tag_list)

        train_dataset = self._generate_dataset(sentences, labels)
        self.model = self._create_model(
            len(self.tag_map),
            len(self.vocab),
            self.embedding_dim,
            self.bidirectional_lstms,
            self.num_lstm_layers,
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.01),
            loss=self._masked_loss,
            metrics=[self._masked_accuracy],
        )

        if validation_data:
            val_dataset = self._generate_dataset(validation_data[0], validation_data[1])
            self.model.fit(
                train_dataset.batch(batch_size),
                validation_data=val_dataset.batch(batch_size),
                epochs=epochs,
            )
        else:
            self.model.fit(train_dataset.batch(batch_size), epochs=epochs)

        print("\nModel Summary:")
        self.model.summary()

        return self

    def predict(
        self,
        sentences: Union[str, List[str]],
        labels: Optional[List[List[str]]] = None,
    ) -> Union[List[str], List[List[str]]]:
        """
        Predicts the labels for the given sentences.

        Args:
        sentences (Union[str, List[str]]): Sentences to predict.
        labels (Optional[List[List[str]]]): True labels for accuracy evaluation.

        Returns:
        Union[List[str], List[List[str]]]: Predicted labels.
        """
        sentence_vectorized = self.sentence_vectorizer(sentences)
        if isinstance(sentences, str):
            sentence_vectorized = tf.expand_dims(sentence_vectorized, axis=0)

        model_outputs = self.model(sentence_vectorized)
        outputs_argmax = np.argmax(model_outputs, axis=-1)

        tag_map_keys = list(self.tag_map.keys())
        pred = []

        for sentence_indices in outputs_argmax:
            sentence_pred = [tag_map_keys[tag_idx] for tag_idx in sentence_indices]
            pred.append(sentence_pred)

        if isinstance(sentences, str):
            pred = pred[0]

        if labels:
            y_true = self._label_vectorizer(labels)
            print(f"The model's accuracy on the provided test set is: {self._masked_accuracy(y_true, model_outputs).numpy():.4f}")

        return pred

    def _get_sentence_vectorizer(
        self, sentences: List[str]
    ) -> Tuple[tf.keras.layers.TextVectorization, List[str]]:
        """
        Creates a TextVectorization layer and adapts it to the sentences.

        Args:
        sentences (List[str]): List of sentences.

        Returns:
        Tuple[tf.keras.layers.TextVectorization, List[str]]: Vectorizer and vocabulary.
        """
        sentence_vectorizer = tf.keras.layers.TextVectorization(standardize=None)
        sentence_vectorizer.adapt(sentences)
        vocab = sentence_vectorizer.get_vocabulary()

        return sentence_vectorizer, vocab

    def _get_tags(self, labels: List[List[str]]) -> List[str]:
        """
        Extracts unique tags from the label sequences.

        Args:
        labels (List[List[str]]): List of label sequences.

        Returns:
        List[str]: List of unique tags.
        """
        tag_set = {tag for label in labels for tag in label}
        tag_list = sorted(list(tag_set))
        return tag_list

    def _make_tag_map(self, tags: List[str]) -> dict:
        """
        Creates a mapping from tags to indices.

        Args:
        tags (List[str]): List of tags.

        Returns:
        dict: Mapping from tags to indices.
        """
        return {tag: i for i, tag in enumerate(tags)}

    def _label_vectorizer(self, labels: List[List[str]]) -> np.ndarray:
        """
        Converts labels to their corresponding indices.

        Args:
        labels (List[List[str]]): List of label sequences.

        Returns:
        np.ndarray: Padded array of label indices.
        """
        label_ids = [[self.tag_map[token] for token in element] for element in labels]
        label_ids = tf.keras.utils.pad_sequences(
            sequences=label_ids, padding="post", value=-1
        )
        return label_ids

    def _generate_dataset(
        self, sentences: List[str], labels: List[List[str]]
    ) -> tf.data.Dataset:
        """
        Generates a TensorFlow dataset from sentences and labels.

        Args:
        sentences (List[str]): List of sentences.
        labels (List[List[str]]): List of label sequences.

        Returns:
        tf.data.Dataset: TensorFlow dataset.
        """
        sentences_ids = self.sentence_vectorizer(sentences)
        labels_ids = self._label_vectorizer(labels)
        dataset = tf.data.Dataset.from_tensor_slices((sentences_ids, labels_ids))
        return dataset

    def _masked_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the masked loss for the model.

        Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.

        Returns:
        tf.Tensor: Computed loss.
        """
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, ignore_class=-1
        )
        loss = loss_fn(y_true, y_pred)
        return loss

    def _masked_accuracy(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the masked accuracy for the model.

        Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.

        Returns:
        tf.Tensor: Computed accuracy.
        """
        y_true = tf.cast(y_true, tf.float32)
        mask = tf.cast(y_true != -1, tf.float32)

        y_pred_class = tf.cast(tf.math.argmax(y_pred, axis=-1), tf.float32)
        matches_true_pred = tf.cast(tf.equal(y_true, y_pred_class), tf.float32) * mask

        masked_acc = tf.reduce_sum(matches_true_pred) / tf.reduce_sum(mask)
        return masked_acc

    def _create_model(
        self,
        len_tags: int,
        vocab_size: int,
        embedding_dim: int,
        bidirectional_lstms: bool,
        num_lstm_layers: int,
    ) -> tf.keras.Model:
        """
        Creates the LSTM model for NER.

        Args:
        len_tags (int): Number of unique tags.
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of word embeddings.
        bidirectional_lstms (bool): Whether to use bidirectional LSTMs.
        num_lstm_layers (int): Number of LSTM layers.

        Returns:
        tf.keras.Model: The created model.
        """
        model = tf.keras.Sequential(name="sequential")
        model.add(tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, mask_zero=True))

        for _ in range(num_lstm_layers):
            if bidirectional_lstms:
                model.add(tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(embedding_dim, return_sequences=True)
                ))
            else:
                model.add(tf.keras.layers.LSTM(embedding_dim, return_sequences=True))

        model.add(tf.keras.layers.Dense(len_tags, activation=tf.nn.log_softmax))
        return model