# Frozen RearClear bottleneck diagnosis

Source `093aaa77a0123138e52f51f576d54e6ae95b36b38929056f8a6a6a22215740dc`
was replayed for exactly 82 already-open track-1 actions and 175 track-4 actions.
All actions and post-step speeds matched their preserved traces. This is a
camera/control diagnosis, not a fresh lap result. No holdout was opened.

Track 1's active right pass (inputs 56–61) drives physical center offset from
-0.3 m to +5.26 m. At input 59 the prospective body corner support is only
0.556 m; at 60 it is zero. The pass excludes the clear-road arc correction.
Input 62 still begins with four wheels on road, but the camera origin component
has only three pixels and two rows, correctly rejecting road confidence. Three
then two wheels remain on road after inputs 62/63. Recovery braking at 62–65
reduces speed to 16.854 m/s. Spin memory is zero and slip before rejection is
about 0.1–1.3 degrees. The initial cause is pass geometry; releasing recovery
throttle would not correct that geometry.

Track 4's later circle is a real future hazard on the supported ridge, not a
disconnected false positive. At input 132 its camera forward distance is
27.631 m, but its closest ridge location is 56.309 m along the bend. Circle to
ridge gap is 1.110 m. An x(y) row-based obstacle phase therefore interrupts a
supported bend too early. Inputs 133–135 have the same distinction; prior pass
is inactive until the row route selects another pass at input 135. Supported
ridge disappears at 136/137 as the vehicle approaches the edge. Later row
extrapolation has roughly 13.42 m road-center error at y=20 m (input 140),
whereas the supported ridge's earlier maximum error at 138 is 0.634 m.
The vehicle subsequently reaches 3/2 road wheels at 142/143, then rejects an
isolated two-pixel origin component at 144. Recovery lasts through 169 with a
minimum 1.755 m/s at 165, without contacts or active spin memory.

The proposed next experiment is restricted to a far-along-ridge circle with a
known stopping prefix, preserving immediate circle avoidance. Static ridge
support alone does not establish support for the emitted turn: inputs 134/135
have insufficient prospective body support. The new deferral experiment must
reject those observations and act earlier at 132/133. Track 1's near-pass
failure remains a separate unresolved mechanism.
