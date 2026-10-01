# NN material workflow

## Course context

Confirmed by Simone on September 29, 2026: NN is co-taught with Danilo Comminiello,
with alternating topics. Danilo covers linear algebra and optimization in the
week of September 28; Simone plans linear models, possibly MLPs, and the first
PyTorch labs for the week of October 5. This is dated planning, not a recurring
schedule or evidence that topics were actually delivered.

Students generally have prior neural-network exposure from a mandatory ML course.
Basic PyTorch exposure is expected but unverified. Keep the core progression
self-contained; use optional refreshers when needed. Advanced additions remain open for discussion. On September 29 Simone selected
the complete existing linear-model and MLP lectures for a template-only port
(with notation adaptation), and PT01/PT02 unchanged for the first labs.

## Discuss and choose

For brainstorming, inspect the relevant NNDS source and current NN material, then
explain reusable content, prerequisites and possible extensions in conversation.
Make the time cost and connection to the core topic clear. Distinguish research
questions from established results and check primary sources for research claims.
Do not turn tentative ideas into approved slides or a settled syllabus.

Before a first port, agree on the topic boundary, available lecture/lab time,
Danilo’s coverage, and which additions earn their place. Reuse NNDS’s review and
full-slide visual QA practices; its course-specific assessment, JAX plans, visual
theme, lecture numbers and public deployment are not NN defaults.

## Port selected material

Resolve the NNDS checkout through the vault Repository Registry and read its
`AGENTS.md` and `NNDS_REFRESH.md`. Identify the authoritative source and distinguish
committed material from working changes or draft notebooks. Do not include another
session’s uncommitted material without an explicit decision.

For each agreed unit:

1. Record the exact NNDS commit, source paths, relevant slide titles or sections,
   target NN paths and supporting assets in `PORTS.md`. Record which source
   content is preserved, adapted, omitted with agreement, or deferred.
2. Port the selected slide source into the NN template. Reference unchanged
   figures, native sources and scripts through the pinned `upstream/nnds` Git
   submodule; do not duplicate them. Copy only assets requiring an NN-specific
   modification into `latex/images/nn/`, recording their source and adaptation.
   Preserve attribution; map required macros explicitly to the NN theme. Avoid
   importing the NNDS preamble, branding or CI wholesale.
   NN uses conventional component subscripts and named position vectors, not
   NNDS's colored square-bracket indexing, which was never introduced here.
3. Keep a linear prerequisite chain across both teachers’ lectures. A prerequisite
   planned for Danilo is not automatically a prerequisite already taught.
4. Implement only the agreed changes, preserving instructor edits and intentional
   NN differences. Review additions in the surrounding course voice and template.
5. Compile and inspect every changed full slide at presentation scale; check
   equations, margins, figures, citations and accidental omissions. Record what
   was actually checked and any unresolved issues with the port.

## Update existing ports

At a requested maintenance pass, compare the recorded source commit with a chosen
new NNDS commit for the mapped files and assets. Review that upstream diff against
the current NN version; apply compatible corrections selectively. Preserve NN
extensions and both teachers’ edits. Do not blindly copy whole files, merge entire
repositories, or silently sync in either direction. Report conflicting teaching
choices for discussion. Update the recorded upstream revision only after each
change is incorporated or explicitly retained/deferred, with the reason.

The submodule is pinned, not a live link to NNDS main. On a new checkout run
`git submodule update --init upstream/nnds`. Advance its recorded commit only in
a requested update after checking all mapped lecture assets. Keep `PORTS.md`
consistent with the selected revision; do not use `git submodule update --remote`
as an automatic build step. Shared lab links in `LABS.md` follow NNDS main
independently of this lecture-asset pin.

For NN-only material, check the touched APIs, references and runnable examples as
part of the requested revision. There is no background updater or scheduled job.

## PyTorch labs

Current decision (October 1): maintain PT01 and PT02 as one shared set for all
courses in the NNDS repository. `LABS.md` links the current source on NNDS main;
do not create course-specific notebook copies. The directory keeps its historical
`drafts` name for stable links. The notebooks may grow with optional sections,
which Simone can select during lectures. Item 25 (tensor semantics) is the first
selected addition; further slide additions are paused for now.
Keep a runnable core and make advanced experiments separable so different levels
can work from the same lab. Experiments should state a question, controlled change,
measurements and interpretation; extra framework syntax alone is not extra depth.

For an approved lab, preserve starter/solution separation, seed experiments where
practical, record dependencies and device assumptions, and execute the solution
from a fresh kernel. Check tensor shapes, loss/target compatibility, gradient
handling, splits and evaluation behavior. Report execution limitations explicitly.

## Publication

The repository is intended to remain private. Desktop commits and pushes follow
the vault’s standing approval and explicit file staging; hosted turns leave Git
to their completion host and report exact changed paths. Creating public links,
deploying a site, inviting collaborators or changing repository visibility needs
its own instruction. This setup grants no automatic synchronization or messages.
