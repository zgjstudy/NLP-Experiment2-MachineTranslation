import tensorflow as tf

import config
from utils import create_masks


class Translator:
    def __init__(self, tokenizer_zh, tokenize_en, model):
        self.tokenizer_zh = tokenizer_zh
        self.tokenizer_en = tokenize_en
        self.model = model

    def encode_zh(self, zh):
        tokens = self.tokenizer_zh.tokenize(zh)
        ids = self.tokenizer_zh.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
        return ids

    def predict(self, zh_sentence):
        zh_sentence = self.encode_zh(zh_sentence)
        encoder_input = tf.expand_dims(zh_sentence, 0)

        decoder_input = [self.tokenizer_en.vocab_size]
        result = tf.expand_dims(decoder_input, 0)

        for i in range(config.max_seq_len):
            target_mask, src_mask = create_masks(encoder_input, result)

            predictions = self.model(encoder_input, result, target_mask, src_mask, False)
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            if tf.equal(predicted_id, self.tokenizer_en.vocab_size + 1):
                break

            result = tf.concat([result, predicted_id], axis=-1)

        return tf.squeeze(result, axis=0)

    def beam_search(self, zh_ids, beam_size):
        encoder_input = tf.expand_dims(zh_ids, 0)

        decoder_input = [self.tokenizer_en.vocab_size]

        target_mask, src_mask = create_masks(encoder_input, tf.expand_dims(decoder_input, 0))
        predictions = self.model(encoder_input, tf.expand_dims(decoder_input, 0), target_mask, src_mask, False)
        predictions = tf.math.log(tf.nn.softmax(predictions[0, -1, :]))

        outputs = [tf.concat([decoder_input, [t]], axis=0) for t in tf.argsort(predictions)[-beam_size:]]  # 候选输出
        scores = [predictions[t] for t in tf.argsort(predictions)[-beam_size:]]  # 候选得分

        for i in range(config.max_seq_len):
            _outputs = []
            _scores = []
            for j in range(beam_size):
                if outputs[j][-1] == self.tokenizer_en.vocab_size + 1:  # 已经到结尾结果的不再搜索
                    _outputs.append(outputs[j])
                    _scores.append(scores[j])
                    continue

                target_mask, src_mask = create_masks(encoder_input, [outputs[j]])
                predictions = self.model(encoder_input, tf.expand_dims(outputs[j], 0), target_mask, src_mask, False)
                predictions = tf.math.log(tf.nn.softmax(predictions[0, -1, :]))

                # 将生成的结果中前 top_k 好的添加到候选
                _outputs.extend([tf.concat([outputs[j], [t]], axis=0) for t in tf.argsort(predictions)[-beam_size:]])
                # 加上本次生成的得分
                _scores.extend([scores[j] + predictions[t] for t in tf.argsort(predictions)[-beam_size:]])

            # 在得到的至多 top_k * top_k 个结果中选择最好的 top_k 个
            _arg_top_k = tf.argsort(_scores)[-beam_size:]
            outputs = [_outputs[t] for t in _arg_top_k]
            scores = [_scores[t] for t in _arg_top_k]

        # length penalty
        def length_penalty(seq):
            return ((5 + len(seq)) ** config.lp_alpha) / ((5 + 1) ** config.lp_alpha)
        scores = [scores[i] / length_penalty(o) for i, o in enumerate(outputs)]

        return outputs[tf.argmax(scores)]

    def translate(self, sentence, beam_size=config.beam_size):
        zh_ids = self.encode_zh(sentence)
        result = self.beam_search(zh_ids, beam_size)
        pred_sentence = self.tokenizer_en.decode([i for i in result if i < self.tokenizer_en.vocab_size])
        return pred_sentence
