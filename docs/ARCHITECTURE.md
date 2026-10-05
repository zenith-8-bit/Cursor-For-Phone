# Architecture

## Responsibility split

### Android AccessibilityService

Owns:

- current UI tree
- semantic element IDs
- node lookup
- clicks
- long presses
- text entry
- scrolling
- global navigation
- app launching
- screen revision

### Python controller

Owns:

- task
- user memory
- Qwen prompt
- action policy
- history
- retry/loop protection
- interactive clarification
- terminal visualization

### Qwen

Only chooses bounded semantic actions.

It does not receive a shell and does not receive arbitrary executable commands.

## Why this is better than screenshot + OCR for the first version

The accessibility tree already contains:

- text
- content descriptions
- resource IDs
- clickable/editable/scrollable flags
- bounds
- checked/selected/focused state
- hierarchy

The model therefore reasons over structured UI semantics instead of trying to reconstruct them from pixels.

A later vision fallback can consume screenshots only when the accessibility tree is insufficient.
