# CNN continuation into sequences and text

> [!note] Written by Codex (2026-10-14)

## Agreed scope

Simone approved a continuation from the remaining convolution variants into
sequences, text representation, prediction and generation. He asked to shorten
the pointwise explanation because positionwise MLPs were already discussed, then
approved removing the audio branch and dilation to streamline the lecture.
The agreed sequence uses one running text example and retains sequence padding.
This is the rendered candidate requested by “Ok let's see how it comes out”.

The current baseline is `c6f548b`, `courses/nnds/Lecture_CNNs.tex`. The imported
material comes from `NNDS_2024/7_CNN_Beyond_Images/Lecture_7_CNNs_beyond_images.tex`.
The source and native figure hashes are recorded in the existing CNN source map.
The accepted earlier content remains the baseline. No shared styles are edited.

## Narrative and dependencies

| Candidate slide | What it uses and explains |
| --- | --- |
| Spatial mixing and channel mixing | Recall the shared 1x1 channel map in one sentence. Retain depthwise filtering, distinguish depthwise from depthwise-separable convolution, and keep the group-convolution aside. |
| Temporal sequences | Reuse the sensor example and annotated shape. The ordered axis has length n; each position has c features. |
| 1D convolutional layers | Adapt the preceding local tensor and shared affine-map definition to one ordered axis. Correct output length, weight orientation and padding assumptions. |
| Handling text data | Preserve the tokenization/embedding progression. Introduce the running example “Check this out”. |
| Text tokenization | Retain the original comparison figure. Fold the compact subword/BPE explanation into its caption. |
| Dense embeddings | Introduce E~(V,d), token IDs (2,16,3) from the original embedding figure, and lookup rows forming X~(3,d). Retain token-level one-hot selection, joint training and the original bias caution/citation. |
| Visualizing the embedding procedure | Retain the original figure and caption. The running sentence and illustrative IDs match this figure. |
| Sequence padding | Combine the source definition and stacking example. Keep the batch shape and explain the validity mask for padded positions. |
| Two tasks on a sequence | Use pooled features for sequence classification or one vocabulary prediction per position. The latter motivates the restriction on future inputs. |
| Causal convolutions | Preserve the causal definition and original figure. Clarify that an output may read its current input and past inputs, while future weights are masked. |
| Training a next-token predictor | Show shifted word tokens/IDs and per-position cross-entropy. Teacher forcing supplies true input tokens; causality permits parallel predictions without target leakage. |
| Autoregressive models | Reuse the generate/append/repeat explanation with the running sentence. Contrast parallel training with sequential sampling. |
| Autoregressive generation | Retain the original generic sequence diagram. Its caption distinguishes generic sequence values from text token IDs. Close with the question about input-dependent interactions that motivates later attention. |

The continuation has 13 content slides, with printed numbers 24 to 36. They are
on PDF pages 28, 30–31, 33–37 and 39–43. The complete candidate has 37 explicit
frames and 44 PDF pages, including seven dividers. The original reading frame is
preserved verbatim at the end.

Notation follows the preceding CNN section: kernel width s=2k+1, neighborhood
radius k, shared W^T, output bias and activation inside each convolutional layer.
For 1D, W has shape (sc,c') and H has shape (n,c') with stride 1 and k zeros at each
end. The depthwise kernel K has shape (s,s,c), with channel index a and offsets
u,v. Its definition includes its activation and bias.

For text, V is vocabulary size, d is embedding dimension, z_i is an integer token
ID, and E is learned floating-point embedding data. Vocabulary probabilities p_i
predict z_(i+1). The loss formula counts valid unpadded positions; padded positions
are excluded with the validity mask. The causal restriction j<=i is a separate
condition. The retained generic generation figure keeps its original one-based
x labels; the new formal sequence equations use zero-based positions and z for
token IDs. No figure labels have been overpainted or redrawn.

## Coverage of the remaining CNN slides

| Original slide | Status | Result |
| --- | --- | --- |
| 1x1 convolutions | substantial edit | Fold the definition into the one-sentence pointwise recap on Spatial mixing and channel mixing, as approved. |
| Depth-wise convolutional layer | substantial edit | Retain independent spatial filtering, its explicit sum, the depthwise-separable composition and the group aside on the combined slide. Correct indexing and include activation/bias to match the accepted layer convention. |
| Further readings | move | Preserve the entire source frame and Chapter 7 recommendation verbatim at the new end. |

## Coverage of the original next lecture

Each original frame has one status. The audio/dilation cuts and compressed
text-prediction sequence were approved in the planning conversation before this
candidate was implemented.

| Original frame | Status | Result |
| --- | --- | --- |
| Lecture 7 title | remove | Keep the current lecture's existing title; do not introduce another numbered lecture. |
| About this lecture | substantial edit | Carry its extension-to-sequences motivation into Temporal sequences and its attention handoff into the closing generation slide. Omit the unsupported categorical claim that CNNs are unsuitable for long dependencies. |
| Temporal sequences | light edit | Retain the sensor example and annotated axes. Correct the inconsistent n/t length notation and follow the current X convention. |
| 1D convolutional layers | light edit | Retain the local affine-plus-activation definition. Correct H's length, use W^T, show full feature slices and qualify padding/equivariance. |
| Masked convolutions | substantial edit | Keep the causal definition on Causal convolutions. Explain the future-weight mask using the original figure instead of the source's ambiguous matrix-row/time-index mask formula. |
| Causal convolutions | move | Preserve the figure on the combined definition slide. Correct its caption to include the current input as well as past inputs. |
| Forecasting | substantial edit | Keep the pooled versus per-position prediction distinction and shifted-target construction. Use the agreed text task with a vocabulary head and cross-entropy instead of a separate continuous-valued MSE example. |
| Forecasting with non-causal vs. causal models | remove | The task comparison, causal figure and explicit shifted text targets carry this distinction in the compressed sequence. |
| Autoregressive models | substantial edit | Keep the generate/append/repeat mechanism and teacher forcing, using the running text example. The mixed-prediction training variant and extrapolation aside are deferred. |
| An example (autoregressive matrices) | remove | Use the agreed single text walkthrough and original generation diagram. |
| Autoregressive generation | move | Preserve the original figure and adapt its caption to the text task. |
| Examples of audio classification | remove | Approved audio cut. |
| Channels and sampling rate | remove | Approved audio cut. |
| Spectrogram classification | remove | Approved audio cut. |
| Sequence padding | substantial edit | Keep the definition and merge the following stacking example. Use embedded text dimensions and explain valid-position masks. |
| An example (sequence padding) | move | Keep its stacking construction and shape on Sequence padding. |
| Can we handle the full waveform? | remove | Approved audio/dilation cut. |
| Visualization of a dilated 1D convolution | remove | Approved dilation cut. |
| WaveNet | remove | Approved audio/dilation cut. |
| Applying convolutive networks to text processing | substantial edit | Retain topic/sentiment classification on Two tasks on a sequence, alongside the agreed next-token task. The application catalogue is compressed. |
| Handling text data | light edit | Preserve the two-operation explanation and resulting matrix. Correct grammar, use first-person plural and distinguish integer IDs from embedding features. |
| Text tokenization | move | Preserve the original figure; combine its caption with the compact BPE explanation. |
| The importance of tokenization | substantial edit | Keep the learned-subword/BPE connection in the tokenization caption. Omit the dated Unicode statistic and link-only demo detour. |
| One-hot encoding | substantial edit | Retain token-level one-hot selection as the equivalent of embedding lookup. Bag-of-words is deferred under the approved scope. |
| Dense embeddings | substantial edit | Preserve learned lookup and joint optimization, with a concrete matrix/table example and IDs from the retained figure. |
| Visualizing the embedding procedure | move | Preserve the complete figure and its original caption. |
| Custom embeddings in Keras | remove | Do not carry the old framework-code slide into this compact narrative. Implementation remains for the PyTorch lab. |
| Dense vs. sparse embeddings | substantial edit | Keep the one-hot selection comparison and bias caution/citation on Dense embeddings. Defer analogy/projector material. |
| Pre-trained text embeddings | remove | Defer the historical catalogue, as proposed and accepted. |

## Figures and verification

Four existing PDF figures and their native XML/Draw.io sources were copied into
`courses/nnds/assets/images/cnn/sequences/`: tokenization, embedding lookup, causal
convolution and autoregressive generation. Every copied file matches its source
bytes. All 14 CNN figure exports match their recorded hashes. Original assets
remain untouched. The compact continuation is LaTeX throughout.

The build and visual review cover all 13 candidate content slides. Local spacing
and figure dimensions were adjusted after full-slide rendering, without changing
the shared template or body font size. The continuation has no overfull box warnings; the title retains its pre-existing
warning. Mathematical checks cover embedding lookup
versus one-hot selection, the 1D neighborhood/affine operation, depthwise channel
independence, causal independence from future values, and shifted targets with
masked loss. These are arithmetic checks, not execution of a trained model.

The earlier source is byte-identical up to Other notable types of convolutions.
A full render comparison of the preceding 26 PDF pages confirms that the title
and all 22 earlier content slides are pixel-identical. Only the three divider
progress bars differ because the total number of frames increased. The original
Further readings frame is also byte-identical.

Changed prose received the humanizer pass against the preceding source voice.
The IDs and sampled continuation are illustrative, as in the source embedding
figure; no claim of real tokenizer output or model execution is made. Source
references checked include [PyTorch Conv1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html),
[PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html),
and the [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary).
The original embedding-bias citation is Bolukbasi et al. (2016), NeurIPS.

This remains an instructor-review candidate. The narrative and cuts are agreed;
acceptance of the rendered wording and layout remains open. No tracked download
PDF is exported, and no HTML, notebook, deep-network lecture or course-site
content is changed by this source checkpoint.


## Instructor correction and subword additions, 2026-10-14

This revision starts from Simone's own edits in `8b6c8a1`. He requested restoring
the boxed causal-model definition and mathematical causal-convolution definition,
and adding one or two slides on subword embeddings and BPE. The earlier candidate
had compressed the definition into unboxed prose and replaced the masked-weight
formula with a diagram. That compression removed original mathematical content
without a specific approval and is corrected here.

These are three added frames and one local move of the existing definition:

| Slide | Requested change and prerequisites |
| --- | --- |
| Subword embeddings | Add after Text tokenization. Use the vocabulary, IDs and embedding matrix already introduced on Handling text data. Show illustrative playing/played splits sharing the play token's embedding, explain unseen-word coverage through known pieces, and distinguish learned pieces from linguistic word parts. |
| Byte pair encoding | Add immediately after Subword embeddings. Explain character-based initialization, repeated most-frequent adjacent-pair merges, target vocabulary size and reuse of fixed ordered rules. A toy corpus has hug x5, hugs x3 and pug x2. Its first two merges have counts 10 and 8. Explain the byte-level variant and its 256-byte alphabet. |
| Masked convolutions | Restore immediately before the causal diagram. Keep the boxed causal-model condition j<=i, the explicit masked affine-plus-activation equation and its masked-weight annotation. Use the accepted W^T orientation and bias/activation convention. Define mask entries by window offset, avoiding the original conflation of matrix columns with absolute sequence positions. |
| Causal convolutions | Move only the unboxed definition onto Masked convolutions. Keep the original figure, dimensions, caption and final explanation. |

For a row-major neighborhood vectorization, the mask M has shape (sc,c'). Its
entry ((u+k)c+a,q) is 1 for u<=0 and 0 for u>0, for every input feature a and output
channel q. The fixed mask and W have identical shapes, and (W elementwise M)^T
maps the sc flattened input features to c' outputs. Bias and pointwise activation
remain inside the convolutional layer. The current input is allowed because the
classifier at i predicts the next token. Token IDs retain Simone's edited
convention; this revision does not change that slide or its embedding notation.

Sources are [Sennrich et al. (ACL 2016)](https://aclanthology.org/P16-1162/)
and the [Hugging Face BPE explanation](https://huggingface.co/learn/llm-course/en/chapter6/5),
linked directly on the added slides. The playing/played split and toy word
frequencies are illustrative, not output from a deployed tokenizer. BPE counts
are recomputed after each merge. The second merge is (h,ug), rather than (h,u).

The revised continuation has 16 content slides (printed 24-39). The complete
deck has 40 explicit frames and 46 PDF pages, including six dividers. All 35 other
existing titled frames are byte-identical to 8b6c8a1, as is the cover and all
surrounding source outside the explicitly changed anchors. No figures or styles
are changed. The mathematical definition is moved, rather than removed.

All five NNDS decks build. Full-slide visual checks cover the three additions and
the affected causal diagram at presentation resolution, with the existing body
font retained. The new frames have no overfull warnings. The title and Handling
text data retain their baseline warnings. Numerical checks confirm the two BPE
merge counts, agreement between the mask equation and a past-only convolution,
and independence from perturbations to future inputs. Changed prose received
the humanizer pass against the surrounding slides. The source and rendered
preservation checks use 8b6c8a1 as the baseline. The cover and 33 unchanged content
frames match the baseline pixels after excluding the updated footer numbers.
Dense embeddings and Autoregressive generation have small rasterization
differences, with identical text and fonts and at most 0.34 pt of text-bounding-box
difference. Both were inspected in full at 1600 px; their layout and content are
preserved. Divider progress bars reflect the increased total slide count.

This remains a rendered instructor-review candidate. The source checkpoint does
not export tracked student PDFs or change the public site.
