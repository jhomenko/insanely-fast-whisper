Whisper
Overview
The Whisper model was proposed in Robust Speech Recognition via Large-Scale Weak Supervision by Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever.

The abstract from the paper is the following:

We study the capabilities of speech processing systems trained simply to predict large amounts of transcripts of audio on the internet. When scaled to 680,000 hours of multilingual and multitask supervision, the resulting models generalize well to standard benchmarks and are often competitive with prior fully supervised results but in a zeroshot transfer setting without the need for any finetuning. When compared to humans, the models approach their accuracy and robustness. We are releasing models and inference code to serve as a foundation for further work on robust speech processing.

Tips:

The model usually performs well without requiring any finetuning.
The architecture follows a classic encoder-decoder architecture, which means that it relies on the generate() function for inference.
One can use WhisperProcessor to prepare audio for the model, and decode the predicted ID’s back into text.
This model was contributed by Arthur Zucker. The Tensorflow version of this model was contributed by amyeroberts. The original code can be found here.

WhisperConfig
class transformers.WhisperConfig
<
source
>
( vocab_size = 51865num_mel_bins = 80encoder_layers = 6encoder_attention_heads = 4decoder_layers = 6decoder_attention_heads = 4decoder_ffn_dim = 1536encoder_ffn_dim = 1536encoder_layerdrop = 0.0decoder_layerdrop = 0.0decoder_start_token_id = 50257use_cache = Trueis_encoder_decoder = Trueactivation_function = 'gelu'd_model = 256dropout = 0.0attention_dropout = 0.0activation_dropout = 0.0init_std = 0.02scale_embedding = Falsemax_source_positions = 1500max_target_positions = 448pad_token_id = 50256bos_token_id = 50257eos_token_id = 50256tie_word_embeddings = Truesuppress_tokens = Nonebegin_suppress_tokens = [220, 50256]**kwargs )

Expand 28 parameters
Parameters

vocab_size (int, optional, defaults to 51865) — Vocabulary size of the Whisper model. Defines the number of different tokens that can be represented by the decoder_input_ids passed when calling WhisperModel
num_mel_bins (int, optional, defaults to 80) — Number of mel features used per input features. Should correspond to the value used in the WhisperProcessor class.
encoder_layers (int, optional, defaults to 6) — Number of encoder layers.
decoder_layers (int, optional, defaults to 6) — Number of decoder layers.
encoder_attention_heads (int, optional, defaults to 4) — Number of attention heads for each attention layer in the Transformer encoder.
decoder_attention_heads (int, optional, defaults to 4) — Number of attention heads for each attention layer in the Transformer decoder.
encoder_ffn_dim (int, optional, defaults to 1536) — Dimensionality of the “intermediate” (often named feed-forward) layer in encoder.
decoder_ffn_dim (int, optional, defaults to 1536) — Dimensionality of the “intermediate” (often named feed-forward) layer in decoder.
encoder_layerdrop (float, optional, defaults to 0.0) — The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556) for more details.
decoder_layerdrop (float, optional, defaults to 0.0) — The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556) for more details.
decoder_start_token_id (int, optional, defaults to 50257) — Corresponds to the ”<|startoftranscript|>” token, which is automatically used when no decoder_input_ids are provided to the generate function. It is used to guide the model`s generation process depending on the task.
use_cache (bool, optional, defaults to True) — Whether or not the model should return the last key/values attentions (not used by all models).
is_encoder_decoder (bool, optional, defaults to True) — Whether the model is used as an encoder/decoder or not.
activation_function (str, optional, defaults to "gelu") — The non-linear activation function (function or string) in the encoder and pooler. If string, "gelu", "relu", "silu" and "gelu_new" are supported.
d_model (int, optional, defaults to 256) — Dimensionality of the layers.
dropout (float, optional, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
attention_dropout (float, optional, defaults to 0.0) — The dropout ratio for the attention probabilities.
activation_dropout (float, optional, defaults to 0.0) — The dropout ratio for activations inside the fully connected layer.
init_std (float, optional, defaults to 0.02) — The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
scale_embedding (bool, optional, defaults to False) — Scale embeddings by diving by sqrt(d_model).
max_source_positions (int, optional, defaults to 1500) — The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
max_target_positions (int, optional, defaults to 448) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
pad_token_id (int, optional, defaults to 50256) — Padding token id.
bos_token_id (int, optional, defaults to 50256) — Begin of stream token id.
eos_token_id (int, optional, defaults to 50257) — End of stream token id.
tie_word_embeddings (bool, optional, defaults to True) — Whether to tie input and output embeddings.
suppress_tokens (List[int], optional) — A list containing the non-speech tokens that will be used by the logit processor in the generate function. NON_SPEECH_TOKENS and NON_SPEECH_TOKENS_MULTI each correspond to the english-only and the multilingual model.
begin_suppress_tokens (List[int], optional, defaults to [220,50256]) — A list containing tokens that will be supressed at the beginning of the sampling process. Initialized as the token for " " (blank_token_id) and the eos_token_id
This is the configuration class to store the configuration of a WhisperModel. It is used to instantiate a Whisper model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Whisper openai/whisper-tiny architecture.

Configuration objects inherit from PretrainedConfig and can be used to control the model outputs. Read the documentation from PretrainedConfig for more information.

Example:

Copied
from transformers import WhisperModel, WhisperConfig

# Initializing a Whisper tiny style configuration
configuration = WhisperConfig()

# Initializing a model from the tiny style configuration
model = WhisperModel(configuration)

# Accessing the model configuration
configuration = model.config
WhisperTokenizer
class transformers.WhisperTokenizer
<
source
>
( vocab_filemerges_filenormalizer_file = Noneerrors = 'replace'unk_token = '<|endoftext|>'bos_token = '<|endoftext|>'eos_token = '<|endoftext|>'pad_token = Noneadd_prefix_space = Falseadd_bos_token = False**kwargs )

Parameters

vocab_file (str) — Path to the vocabulary file.
merges_file (str) — Path to the merges file.
normalizer_file (str, optional, defaults to None) — Path to the normalizer_file file.
errors (str, optional, defaults to "replace") — Paradigm to follow when decoding bytes to UTF-8. See bytes.decode for more information.
unk_token (str, optional, defaults to "<|endoftext|>") — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
bos_token (str, optional, defaults to "<|endoftext|>") — The beginning of sequence token.
eos_token (str, optional, defaults to "<|endoftext|>") — The end of sequence token.
add_prefix_space (bool, optional, defaults to False) — Whether or not to add an initial space to the input. This allows to treat the leading word just as any other word.
add_bos_token (bool, optional, defaults to False) — Whether or not to add an initial <|endoftext|> to the input. This allows to treat the leading word just as any other word.
Construct an Whisper tokenizer.

This tokenizer inherits from PreTrainedTokenizer which contains some of the main methods. Users should refer to the superclass for more information regarding such methods.

build_inputs_with_special_tokens
<
source
>
( token_ids_0token_ids_1 = None )

get_special_tokens_mask
<
source
>
( token_ids_0: typing.List[int]token_ids_1: typing.Optional[typing.List[int]] = Nonealready_has_special_tokens: bool = False ) → List[int]

Parameters

token_ids_0 (List[int]) — List of IDs.
token_ids_1 (List[int], optional) — Optional second list of IDs for sequence pairs.
already_has_special_tokens (bool, optional, defaults to False) — Whether or not the token list is already formatted with special tokens for the model.
Returns

List[int]

A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.


Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer prepare_for_model or encode_plus methods.

create_token_type_ids_from_sequences
<
source
>
( token_ids_0: typing.List[int]token_ids_1: typing.Optional[typing.List[int]] = None ) → List[int]

Parameters

token_ids_0 (List[int]) — The first tokenized sequence.
token_ids_1 (List[int], optional) — The second tokenized sequence.
Returns

List[int]

The token type ids.


Create the token type IDs corresponding to the sequences passed. What are token type IDs?

Should be overridden in a subclass if the model has a special way of building those.

save_vocabulary
<
source
>
( save_directory: strfilename_prefix: typing.Optional[str] = None )

WhisperFeatureExtractor
class transformers.WhisperFeatureExtractor
<
source
>
( feature_size = 80sampling_rate = 16000hop_length = 160chunk_length = 30n_fft = 400padding_value = 0.0**kwargs )

Parameters

feature_size (int, defaults to 80) — The feature dimension of the extracted features.
sampling_rate (int, defaults to 16000) — The sampling rate at which the audio files should be digitalized expressed in Hertz per second (Hz).
hop_length (int, defaults to 160) — Length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients.
chunk_length (int, defaults to 30) — The maximum number of chuncks of sampling_rate samples used to trim and pad longer or shorter audio sequences.
n_fft (int, defaults to 400) — Size of the Fourier transform.
padding_value (float, optional, defaults to 0.0) — Padding value used to pad the audio. Should correspond to silences.
Constructs a Whisper feature extractor.

This feature extractor inherits from WhisperFeatureExtractor which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

This class extracts mel-filter bank features from raw speech using a custom numpy implementation of the Short Time Fourier Transform which should match pytorch’s torch.stft equivalent.

__call__
<
source
>
( raw_speech: typing.Union[numpy.ndarray, typing.List[float], typing.List[numpy.ndarray], typing.List[typing.List[float]]]truncation: bool = Truepad_to_multiple_of: typing.Optional[int] = Nonereturn_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = Nonereturn_attention_mask: typing.Optional[bool] = Nonepadding: typing.Optional[str] = 'max_length'max_length: typing.Optional[int] = Nonesampling_rate: typing.Optional[int] = None**kwargs )

Parameters

raw_speech (np.ndarray, List[float], List[np.ndarray], List[List[float]]) — The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float values, a list of numpy arrays or a list of list of float values.
truncation (bool, optional, default to True) — Activates truncation to cut input sequences longer than max_length to max_length.
pad_to_multiple_of (int, optional, defaults to None) — If set will pad the sequence to a multiple of the provided value.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability

= 7.5 (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.

return_attention_mask (bool, optional) — Whether to return the attention mask. If left to the default, will return the attention mask according to the specific feature_extractor’s default.
What are attention masks?

For WhisperTransoformer models, attention_mask should alwys be passed for batched inference, to avoid subtle bugs.

return_tensors (str or TensorType, optional) — If set, will return tensors instead of list of python integers. Acceptable values are:
'tf': Return TensorFlow tf.constant objects.
'pt': Return PyTorch torch.Tensor objects.
'np': Return Numpy np.ndarray objects.
sampling_rate (int, optional) — The sampling rate at which the raw_speech input was sampled. It is strongly recommended to pass sampling_rate at the forward call to prevent silent errors and allow automatic speech recognition pipeline.
padding_value (float, defaults to 0.0) — The value that is used to fill the padding values / vectors.
Main method to featurize and prepare for the model one or several sequence(s).

WhisperProcessor
class transformers.WhisperProcessor
<
source
>
( feature_extractortokenizer )

Parameters

feature_extractor (WhisperFeatureExtractor) — An instance of WhisperFeatureExtractor. The feature extractor is a required input.
tokenizer (WhisperTokenizer) — An instance of WhisperTokenizer. The tokenizer is a required input.
Constructs a Whisper processor which wraps a Whisper feature extractor and a Whisper tokenizer into a single processor.

WhisperProcessor offers all the functionalities of WhisperFeatureExtractor and WhisperTokenizer. See the call() and decode() for more information.

__call__
<
source
>
( *args**kwargs )

When used in normal mode, this method forwards all its arguments to WhisperFeatureExtractor’s call() and returns its output. If used in the context ~WhisperProcessor.as_target_processor this method forwards all its arguments to WhisperTokenizer’s call(). Please refer to the doctsring of the above two methods for more information.

from_pretrained
<
source
>
( pretrained_model_name_or_path**kwargs )

Parameters

pretrained_model_name_or_path (str or os.PathLike) — This can be either:
a string, the model id of a pretrained feature_extractor hosted inside a model repo on huggingface.co. Valid model ids can be located at the root-level, like bert-base-uncased, or namespaced under a user or organization name, like dbmdz/bert-base-german-cased.
a path to a directory containing a feature extractor file saved using the save_pretrained() method, e.g., ./my_model_directory/.
a path or url to a saved feature extractor JSON file, e.g., ./my_model_directory/preprocessor_config.json. **kwargs — Additional keyword arguments passed along to both from_pretrained() and ~tokenization_utils_base.PreTrainedTokenizer.from_pretrained.
Instantiate a processor associated with a pretrained model.

This class method is simply calling the feature extractor from_pretrained() and the tokenizer ~tokenization_utils_base.PreTrainedTokenizer.from_pretrained methods. Please refer to the docstrings of the methods above for more information.

save_pretrained
<
source
>
( save_directorypush_to_hub: bool = False**kwargs )

Parameters

save_directory (str or os.PathLike) — Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will be created if it does not exist).
push_to_hub (bool, optional, defaults to False) — Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with repo_id (will default to the name of save_directory in your namespace). kwargs — Additional key word arguments passed along to the push_to_hub() method.
Saves the attributes of this processor (feature extractor, tokenizer…) in the specified directory so that it can be reloaded using the from_pretrained() method.

This class method is simply calling save_pretrained() and ~tokenization_utils_base.PreTrainedTokenizer.save_pretrained. Please refer to the docstrings of the methods above for more information.

batch_decode
<
source
>
( *args**kwargs )

This method forwards all its arguments to WhisperTokenizer’s batch_decode(). Please refer to the docstring of this method for more information.

decode
<
source
>
( *args**kwargs )

This method forwards all its arguments to WhisperTokenizer’s decode(). Please refer to the docstring of this method for more information.

WhisperModel
class transformers.WhisperModel
<
source
>
( config: WhisperConfig )

Parameters

config (WhisperConfig) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the from_pretrained() method to load the model weights.
The bare Whisper Model outputting raw hidden-states without any specific head on top. This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

forward
<
source
>
( input_features = Nonedecoder_input_ids = Nonedecoder_attention_mask = Nonehead_mask = Nonedecoder_head_mask = Nonecross_attn_head_mask = Noneencoder_outputs = Nonepast_key_values = Nonedecoder_inputs_embeds = Noneuse_cache = Noneoutput_attentions = Noneoutput_hidden_states = Nonereturn_dict = None ) → transformers.modeling_outputs.Seq2SeqModelOutput or tuple(torch.FloatTensor)

Expand 12 parameters
Parameters

input_features (torch.FloatTensor of shape (batch_size, feature_size, sequence_length)) — Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by loading a .flac or .wav audio file into an array of type List[float] or a numpy.ndarray, e.g. via the soundfile library (pip install soundfile). To prepare the array into input_features, the WhisperFeatureExtractor should be used for extracting the mel features, padding and conversion into a tensor of type torch.FloatTensor. See call()
decoder_input_ids (torch.LongTensor of shape (batch_size, target_sequence_length), optional) — Indices of decoder input sequence tokens in the vocabulary.
Indices can be obtained using WhisperTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are decoder input IDs?

Whisper uses the decoder_start_token_id as the starting token for decoder_input_ids generation. If past_key_values is used, optionally only the last decoder_input_ids have to be input (see past_key_values).

decoder_attention_mask (torch.LongTensor of shape (batch_size, target_sequence_length), optional) — Default behavior: generate a tensor that ignores pad tokens in decoder_input_ids. Causal mask will also be used by default.
If you want to change padding behavior, you should read modeling_whisper._prepare_decoder_attention_mask and modify to your needs. See diagram 1 in the BART paper for more information on the default strategy.

head_mask (torch.Tensor of shape (encoder_layers, encoder_attention_heads), optional) — Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in [0, 1]:
1 indicates the head is not masked,
0 indicates the head is masked.
decoder_head_mask (torch.Tensor of shape (decoder_layers, decoder_attention_heads), optional) — Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in [0, 1]:
1 indicates the head is not masked,
0 indicates the head is masked.
cross_attn_head_mask (torch.Tensor of shape (decoder_layers, decoder_attention_heads), optional) — Mask to nullify selected heads of the cross-attention modules. Mask values selected in [0, 1]:
1 indicates the head is not masked,
0 indicates the head is masked.
encoder_outputs (tuple(tuple(torch.FloatTensor), optional) — Tuple consists of (last_hidden_state, optional: hidden_states, optional: attentions) last_hidden_state of shape (batch_size, sequence_length, hidden_size), optional) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
past_key_values (tuple(tuple(torch.FloatTensor)), optional, returned when use_cache=True is passed or when config.use_cache=True) — Tuple of tuple(torch.FloatTensor) of length config.n_layers, with each tuple having 2 tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head)) and 2 additional tensors of shape (batch_size, num_heads, encoder_sequence_length, embed_size_per_head).
Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

If past_key_values are used, the user can optionally input only the last decoder_input_ids (those that don’t have their past key value states given to this model) of shape (batch_size, 1) instead of all decoder_input_ids of shape (batch_size, sequence_length). decoder_inputs_embeds (torch.FloatTensor of shape (batch_size, target_sequence_length, hidden_size), optional): Optionally, instead of passing decoder_input_ids you can choose to directly pass an embedded representation. If past_key_values is used, optionally only the last decoder_inputs_embeds have to be input (see past_key_values). This is useful if you want more control over how to convert decoder_input_ids indices into associated vectors than the model’s internal embedding lookup matrix.

If decoder_input_ids and decoder_inputs_embeds are both unset, decoder_inputs_embeds takes the value of inputs_embeds.

use_cache (bool, optional) — If set to True, past_key_values key value states are returned and can be used to speed up decoding (see past_key_values).
output_attentions (bool, optional) — Whether or not to return the attentions tensors of all attention layers. See attentions under returned tensors for more detail.
output_hidden_states (bool, optional) — Whether or not to return the hidden states of all layers. See hidden_states under returned tensors for more detail.
return_dict (bool, optional) — Whether or not to return a ModelOutput instead of a plain tuple.
Returns

transformers.modeling_outputs.Seq2SeqModelOutput or tuple(torch.FloatTensor)

A transformers.modeling_outputs.Seq2SeqModelOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (WhisperConfig) and inputs.

last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the decoder of the model.

If past_key_values is used only the last hidden-state of the sequences of shape (batch_size, 1, hidden_size) is output.

past_key_values (tuple(tuple(torch.FloatTensor)), optional, returned when use_cache=True is passed or when config.use_cache=True) — Tuple of tuple(torch.FloatTensor) of length config.n_layers, with each tuple having 2 tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head)) and 2 additional tensors of shape (batch_size, num_heads, encoder_sequence_length, embed_size_per_head).

Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

decoder_hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.

decoder_attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

encoder_last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size), optional) — Sequence of hidden-states at the output of the last layer of the encoder of the model.

encoder_hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.

encoder_attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.


The WhisperModel forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

Copied
from transformers import openai/whisper-tiny, WhisperModel
import torch
from datasets import load_dataset

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = openai/whisper-tiny.from_pretrained("openai/whisper-tiny")
model = WhisperModel.from_pretrained("openai/whisper-tiny")

# audio file is decoded on the fly
inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)
[1, 2, 512]
WhisperForConditionalGeneration
class transformers.WhisperForConditionalGeneration
<
source
>
( config: WhisperConfig )

Parameters

config (WhisperConfig) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the from_pretrained() method to load the model weights.
The Whisper Model with a language modeling head. Can be used for automatic speech recognition. This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

forward
<
source
>
( input_features = Nonedecoder_input_ids = Nonedecoder_attention_mask = Nonehead_mask = Nonedecoder_head_mask = Nonecross_attn_head_mask = Noneencoder_outputs = Nonepast_key_values = Nonedecoder_inputs_embeds = Nonelabels = Noneuse_cache = Noneoutput_attentions = Noneoutput_hidden_states = Nonereturn_dict = None ) → transformers.modeling_outputs.Seq2SeqLMOutput or tuple(torch.FloatTensor)

Expand 13 parameters
Parameters

input_features (torch.FloatTensor of shape (batch_size, feature_size, sequence_length)) — Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by loading a .flac or .wav audio file into an array of type List[float] or a numpy.ndarray, e.g. via the soundfile library (pip install soundfile). To prepare the array into input_features, the WhisperFeatureExtractor should be used for extracting the mel features, padding and conversion into a tensor of type torch.FloatTensor. See call()
decoder_input_ids (torch.LongTensor of shape (batch_size, target_sequence_length), optional) — Indices of decoder input sequence tokens in the vocabulary.
Indices can be obtained using WhisperTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are decoder input IDs?

Whisper uses the decoder_start_token_id as the starting token for decoder_input_ids generation. If past_key_values is used, optionally only the last decoder_input_ids have to be input (see past_key_values).

decoder_attention_mask (torch.LongTensor of shape (batch_size, target_sequence_length), optional) — Default behavior: generate a tensor that ignores pad tokens in decoder_input_ids. Causal mask will also be used by default.
If you want to change padding behavior, you should read modeling_whisper._prepare_decoder_attention_mask and modify to your needs. See diagram 1 in the BART paper for more information on the default strategy.

head_mask (torch.Tensor of shape (encoder_layers, encoder_attention_heads), optional) — Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in [0, 1]:
1 indicates the head is not masked,
0 indicates the head is masked.
decoder_head_mask (torch.Tensor of shape (decoder_layers, decoder_attention_heads), optional) — Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in [0, 1]:
1 indicates the head is not masked,
0 indicates the head is masked.
cross_attn_head_mask (torch.Tensor of shape (decoder_layers, decoder_attention_heads), optional) — Mask to nullify selected heads of the cross-attention modules. Mask values selected in [0, 1]:
1 indicates the head is not masked,
0 indicates the head is masked.
encoder_outputs (tuple(tuple(torch.FloatTensor), optional) — Tuple consists of (last_hidden_state, optional: hidden_states, optional: attentions) last_hidden_state of shape (batch_size, sequence_length, hidden_size), optional) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
past_key_values (tuple(tuple(torch.FloatTensor)), optional, returned when use_cache=True is passed or when config.use_cache=True) — Tuple of tuple(torch.FloatTensor) of length config.n_layers, with each tuple having 2 tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head)) and 2 additional tensors of shape (batch_size, num_heads, encoder_sequence_length, embed_size_per_head).
Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

If past_key_values are used, the user can optionally input only the last decoder_input_ids (those that don’t have their past key value states given to this model) of shape (batch_size, 1) instead of all decoder_input_ids of shape (batch_size, sequence_length). decoder_inputs_embeds (torch.FloatTensor of shape (batch_size, target_sequence_length, hidden_size), optional): Optionally, instead of passing decoder_input_ids you can choose to directly pass an embedded representation. If past_key_values is used, optionally only the last decoder_inputs_embeds have to be input (see past_key_values). This is useful if you want more control over how to convert decoder_input_ids indices into associated vectors than the model’s internal embedding lookup matrix.

If decoder_input_ids and decoder_inputs_embeds are both unset, decoder_inputs_embeds takes the value of inputs_embeds.

use_cache (bool, optional) — If set to True, past_key_values key value states are returned and can be used to speed up decoding (see past_key_values).
output_attentions (bool, optional) — Whether or not to return the attentions tensors of all attention layers. See attentions under returned tensors for more detail.
output_hidden_states (bool, optional) — Whether or not to return the hidden states of all layers. See hidden_states under returned tensors for more detail.
return_dict (bool, optional) — Whether or not to return a ModelOutput instead of a plain tuple.
labels (torch.LongTensor of shape (batch_size, sequence_length), optional) — Labels for computing the language modeling loss. Indices should either be in [0, ..., config.vocab_size] or -100 (see input_ids docstring). Tokens with indices set to -100 are ignored (masked), the loss is only computed for the tokens with labels in [0, ..., config.vocab_size].
Returns

transformers.modeling_outputs.Seq2SeqLMOutput or tuple(torch.FloatTensor)

A transformers.modeling_outputs.Seq2SeqLMOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (WhisperConfig) and inputs.

loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Language modeling loss.

logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

past_key_values (tuple(tuple(torch.FloatTensor)), optional, returned when use_cache=True is passed or when config.use_cache=True) — Tuple of tuple(torch.FloatTensor) of length config.n_layers, with each tuple having 2 tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head)) and 2 additional tensors of shape (batch_size, num_heads, encoder_sequence_length, embed_size_per_head).

Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

decoder_hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.

decoder_attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

encoder_last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size), optional) — Sequence of hidden-states at the output of the last layer of the encoder of the model.

encoder_hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.

encoder_attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.


The WhisperForConditionalGeneration forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

Copied
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
input_features = inputs.input_features

generated_ids = model.generate(inputs=input_features)

transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
transcription
' Mr. Quilter is the apostle of the middle classes, and we are glad to'
TFWhisperModel
class transformers.TFWhisperModel
<
source
>
( *args**kwargs )

Parameters

config (WhisperConfig) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the from_pretrained() method to load the model weights.
The bare Whisper Model outputting raw hidden-states without any specific head on top. This model inherits from TFPreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a tf.keras.Model subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

call
<
source
>
( input_features = Nonedecoder_input_ids = Nonedecoder_attention_mask = Nonedecoder_position_ids = Nonehead_mask = Nonedecoder_head_mask = Nonecross_attn_head_mask = Noneencoder_outputs = Nonepast_key_values = Nonedecoder_inputs_embeds = Noneuse_cache = Noneoutput_attentions = Noneoutput_hidden_states = Nonereturn_dict = Nonetraining = False ) → transformers.modeling_tf_outputs.TFSeq2SeqLMOutput or tuple(tf.Tensor)

Expand 12 parameters
Parameters

input_features (tf.Tensor of shape (batch_size, feature_size, sequence_length)) — Float values of fbank features extracted from the raw speech waveform. Raw speech waveform can be obtained by loading a .flac or .wav audio file into an array of type List[float] or a numpy.ndarray, e.g. via the soundfile library (pip install soundfile). To prepare the array into input_features, the WhisperFeatureExtractor should be used for extracting the fbank features, padding and conversion into a tensor of type tf.Tensor. See call()
decoder_input_ids (tf.Tensor of shape (batch_size, target_sequence_length), optional) — Indices of decoder input sequence tokens in the vocabulary.
Indices can be obtained using SpeechToTextTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are decoder input IDs?

SpeechToText uses the eos_token_id as the starting token for decoder_input_ids generation. If past_key_values is used, optionally only the last decoder_input_ids have to be input (see past_key_values).

decoder_attention_mask (tf.Tensor of shape (batch_size, target_sequence_length), optional) — Default behavior: generate a tensor that ignores pad tokens in decoder_input_ids. Causal mask will also be used by default.
If you want to change padding behavior, you should read modeling_whisper._prepare_decoder_attention_mask and modify to your needs. See diagram 1 in the paper for more information on the default strategy.

head_mask (tf.Tensor of shape (encoder_layers, encoder_attention_heads), optional) — Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in [0, 1]:
1 indicates the head is not masked,
0 indicates the head is masked.
decoder_head_mask (tf.Tensor of shape (decoder_layers, decoder_attention_heads), optional) — Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in [0, 1]:
1 indicates the head is not masked,
0 indicates the head is masked.
cross_attn_head_mask (tf.Tensor of shape (decoder_layers, decoder_attention_heads), optional) — Mask to nullify selected heads of the cross-attention modules. Mask values selected in [0, 1]:
1 indicates the head is not masked,
0 indicates the head is masked.
encoder_outputs (tuple(tuple(tf.Tensor), optional) — Tuple consists of (last_hidden_state, optional: hidden_states, optional: attentions) last_hidden_state of shape (batch_size, sequence_length, hidden_size), optional) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
past_key_values (tuple(tuple(tf.Tensor)), optional, returned when use_cache=True is passed or when config.use_cache=True) — Tuple of tuple(tf.Tensor) of length config.n_layers, with each tuple having 2 tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head)) and 2 additional tensors of shape (batch_size, num_heads, encoder_sequence_length, embed_size_per_head).
Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

If past_key_values are used, the user can optionally input only the last decoder_input_ids (those that don’t have their past key value states given to this model) of shape (batch_size, 1) instead of all decoder_input_ids of shape (batch_size, sequence_length). decoder_inputs_embeds (tf.Tensor of shape (batch_size, target_sequence_length, hidden_size), optional): Optionally, instead of passing decoder_input_ids you can choose to directly pass an embedded representation. If past_key_values is used, optionally only the last decoder_inputs_embeds have to be input (see past_key_values). This is useful if you want more control over how to convert decoder_input_ids indices into associated vectors than the model’s internal embedding lookup matrix.

If decoder_input_ids and decoder_inputs_embeds are both unset, decoder_inputs_embeds takes the value of inputs_embeds.

use_cache (bool, optional) — If set to True, past_key_values key value states are returned and can be used to speed up decoding (see past_key_values).
output_attentions (bool, optional) — Whether or not to return the attentions tensors of all attention layers. See attentions under returned tensors for more detail.
output_hidden_states (bool, optional) — Whether or not to return the hidden states of all layers. See hidden_states under returned tensors for more detail.
return_dict (bool, optional) — Whether or not to return a ModelOutput instead of a plain tuple.
Returns

transformers.modeling_tf_outputs.TFSeq2SeqLMOutput or tuple(tf.Tensor)

A transformers.modeling_tf_outputs.TFSeq2SeqLMOutput or a tuple of tf.Tensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (WhisperConfig) and inputs.

loss (tf.Tensor of shape (n,), optional, where n is the number of non-masked labels, returned when labels is provided) — Language modeling loss.

logits (tf.Tensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

past_key_values (List[tf.Tensor], optional, returned when use_cache=True is passed or when config.use_cache=True) — List of tf.Tensor of length config.n_layers, with each tensor of shape (2, batch_size, num_heads, sequence_length, embed_size_per_head)).

Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be used (see past_key_values input) to speed up sequential decoding.

decoder_hidden_states (tuple(tf.Tensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of tf.Tensor (one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.

decoder_attentions (tuple(tf.Tensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of tf.Tensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (tuple(tf.Tensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of tf.Tensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

encoder_last_hidden_state (tf.Tensor of shape (batch_size, sequence_length, hidden_size), optional) — Sequence of hidden-states at the output of the last layer of the encoder of the model.

encoder_hidden_states (tuple(tf.Tensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of tf.Tensor (one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.

encoder_attentions (tuple(tf.Tensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of tf.Tensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.


The TFWhisperModel forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

Copied
import tensorflow as tf
from transformers import TFWhisperModel, WhisperFeatureExtractor
from datasets import load_dataset

model = TFWhisperModel.from_pretrained("openai/whisper-base")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
inputs = feature_extractor(
    ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="tf"
)
input_features = inputs.input_features
decoder_input_ids = tf.convert_to_tensor([[1, 1]]) * model.config.decoder_start_token_id
last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
list(last_hidden_state.shape)
[1, 2, 512]
TFWhisperForConditionalGeneration
class transformers.TFWhisperForConditionalGeneration
<
source
>
( *args**kwargs )

Parameters

config (WhisperConfig) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the from_pretrained() method to load the model weights.
The Whisper Model with a language modeling head. Can be used for automatic speech recognition. This model inherits from TFPreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a tf.keras.Model subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

call
<
source
>
( input_features = Nonedecoder_input_ids = Nonedecoder_attention_mask = Nonedecoder_position_ids = Nonehead_mask = Nonedecoder_head_mask = Nonecross_attn_head_mask = Noneencoder_outputs = Nonepast_key_values = Nonedecoder_inputs_embeds = Nonelabels = Noneuse_cache = Noneoutput_attentions = Noneoutput_hidden_states = Nonereturn_dict = Nonetraining = False ) → transformers.modeling_tf_outputs.TFSeq2SeqLMOutput or tuple(tf.Tensor)

Expand 13 parameters
Parameters

input_features (tf.Tensor of shape (batch_size, feature_size, sequence_length)) — Float values of fbank features extracted from the raw speech waveform. Raw speech waveform can be obtained by loading a .flac or .wav audio file into an array of type List[float] or a numpy.ndarray, e.g. via the soundfile library (pip install soundfile). To prepare the array into input_features, the WhisperFeatureExtractor should be used for extracting the fbank features, padding and conversion into a tensor of type tf.Tensor. See call()
decoder_input_ids (tf.Tensor of shape (batch_size, target_sequence_length), optional) — Indices of decoder input sequence tokens in the vocabulary.
Indices can be obtained using SpeechToTextTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are decoder input IDs?

SpeechToText uses the eos_token_id as the starting token for decoder_input_ids generation. If past_key_values is used, optionally only the last decoder_input_ids have to be input (see past_key_values).

decoder_attention_mask (tf.Tensor of shape (batch_size, target_sequence_length), optional) — Default behavior: generate a tensor that ignores pad tokens in decoder_input_ids. Causal mask will also be used by default.
If you want to change padding behavior, you should read modeling_whisper._prepare_decoder_attention_mask and modify to your needs. See diagram 1 in the paper for more information on the default strategy.

head_mask (tf.Tensor of shape (encoder_layers, encoder_attention_heads), optional) — Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in [0, 1]:
1 indicates the head is not masked,
0 indicates the head is masked.
decoder_head_mask (tf.Tensor of shape (decoder_layers, decoder_attention_heads), optional) — Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in [0, 1]:
1 indicates the head is not masked,
0 indicates the head is masked.
cross_attn_head_mask (tf.Tensor of shape (decoder_layers, decoder_attention_heads), optional) — Mask to nullify selected heads of the cross-attention modules. Mask values selected in [0, 1]:
1 indicates the head is not masked,
0 indicates the head is masked.
encoder_outputs (tuple(tuple(tf.Tensor), optional) — Tuple consists of (last_hidden_state, optional: hidden_states, optional: attentions) last_hidden_state of shape (batch_size, sequence_length, hidden_size), optional) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
past_key_values (tuple(tuple(tf.Tensor)), optional, returned when use_cache=True is passed or when config.use_cache=True) — Tuple of tuple(tf.Tensor) of length config.n_layers, with each tuple having 2 tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head)) and 2 additional tensors of shape (batch_size, num_heads, encoder_sequence_length, embed_size_per_head).
Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

If past_key_values are used, the user can optionally input only the last decoder_input_ids (those that don’t have their past key value states given to this model) of shape (batch_size, 1) instead of all decoder_input_ids of shape (batch_size, sequence_length). decoder_inputs_embeds (tf.Tensor of shape (batch_size, target_sequence_length, hidden_size), optional): Optionally, instead of passing decoder_input_ids you can choose to directly pass an embedded representation. If past_key_values is used, optionally only the last decoder_inputs_embeds have to be input (see past_key_values). This is useful if you want more control over how to convert decoder_input_ids indices into associated vectors than the model’s internal embedding lookup matrix.

If decoder_input_ids and decoder_inputs_embeds are both unset, decoder_inputs_embeds takes the value of inputs_embeds.

use_cache (bool, optional) — If set to True, past_key_values key value states are returned and can be used to speed up decoding (see past_key_values).
output_attentions (bool, optional) — Whether or not to return the attentions tensors of all attention layers. See attentions under returned tensors for more detail.
output_hidden_states (bool, optional) — Whether or not to return the hidden states of all layers. See hidden_states under returned tensors for more detail.
return_dict (bool, optional) — Whether or not to return a ModelOutput instead of a plain tuple.
labels (tf.Tensor of shape (batch_size, sequence_length), optional) — Labels for computing the language modeling loss. Indices should either be in [0, ..., config.vocab_size] or -100 (see input_ids docstring). Tokens with indices set to -100 are ignored (masked), the loss is only computed for the tokens with labels in [0, ..., config.vocab_size].
Returns

transformers.modeling_tf_outputs.TFSeq2SeqLMOutput or tuple(tf.Tensor)

A transformers.modeling_tf_outputs.TFSeq2SeqLMOutput or a tuple of tf.Tensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (WhisperConfig) and inputs.

loss (tf.Tensor of shape (n,), optional, where n is the number of non-masked labels, returned when labels is provided) — Language modeling loss.

logits (tf.Tensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

past_key_values (List[tf.Tensor], optional, returned when use_cache=True is passed or when config.use_cache=True) — List of tf.Tensor of length config.n_layers, with each tensor of shape (2, batch_size, num_heads, sequence_length, embed_size_per_head)).

Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be used (see past_key_values input) to speed up sequential decoding.

decoder_hidden_states (tuple(tf.Tensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of tf.Tensor (one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.

decoder_attentions (tuple(tf.Tensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of tf.Tensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (tuple(tf.Tensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of tf.Tensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

encoder_last_hidden_state (tf.Tensor of shape (batch_size, sequence_length, hidden_size), optional) — Sequence of hidden-states at the output of the last layer of the encoder of the model.

encoder_hidden_states (tuple(tf.Tensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of tf.Tensor (one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.

encoder_attentions (tuple(tf.Tensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of tf.Tensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.


The TFWhisperForConditionalGeneration forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

Copied
import tensorflow as tf
from transformers import WhisperProcessor, TFWhisperForConditionalGeneration
from datasets import load_dataset

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

inputs = processor(ds[0]["audio"]["array"], return_tensors="tf")
input_features = inputs.input_features

generated_ids = model.generate(inputs=input_features)

transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
transcription
' Mr. Quilter is the apostle of the middle classes, and we are glad to'