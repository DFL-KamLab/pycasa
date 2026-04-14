# API: Motility

Purpose:
Motility transforms tracks into biologically meaningful motion metrics (velocity, linearity, wobble, etc.) using sliding-window calculations over trajectories.

## Public Methods In This Section

- `self.motility.standard_motility_parameters(...)`

## Example

```python
self.motility.standard_motility_parameters(show_progress=True, verbose=True)
motility = self.get_motility()
```

## Output Behavior

- Writes metrics into `casa["motility"]`.
- Updates `casa["meta"]["last_motility"]`.
- Depends on prior tracking output for meaningful results.
