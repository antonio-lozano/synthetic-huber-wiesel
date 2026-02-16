/**
 * PresetPanel – load built-in or custom presets; save current settings as a new preset.
 */
import { useState } from "react";
import { useSimStore, Preset } from "../core/store";

export function PresetPanel() {
  const s = useSimStore();
  const [newPresetName, setNewPresetName] = useState("");

  const allPresets: { preset: Preset; isCustom: boolean; customIndex: number }[] = [
    ...s.presets.map((p) => ({ preset: p, isCustom: false, customIndex: -1 })),
    ...s.customPresets.map((p, i) => ({ preset: p, isCustom: true, customIndex: i })),
  ];

  return (
    <section className="card" role="region" aria-label="Presets">
      <h2>Presets</h2>

      <div className="preset-list">
        {allPresets.map(({ preset, isCustom, customIndex }, idx) => (
          <div key={idx} className="preset-row">
            <button
              type="button"
              className="preset-btn"
              onClick={() => s.applyPreset(preset)}
              aria-label={`Load preset: ${preset.name}`}
              title={`Stimulus: ${preset.stimKind}, Mode: ${preset.responseMode}`}
            >
              {preset.name}
            </button>
            {isCustom && (
              <button
                type="button"
                className="preset-delete"
                onClick={() => s.deleteCustomPreset(customIndex)}
                aria-label={`Delete preset: ${preset.name}`}
                title="Delete"
              >
                ×
              </button>
            )}
          </div>
        ))}
      </div>

      <div className="preset-save-row">
        <input
          type="text"
          placeholder="New preset name…"
          value={newPresetName}
          onChange={(e) => setNewPresetName(e.target.value)}
          aria-label="New preset name"
          maxLength={40}
        />
        <button
          type="button"
          disabled={newPresetName.trim().length === 0}
          onClick={() => {
            s.saveCustomPreset(newPresetName.trim());
            setNewPresetName("");
          }}
          aria-label="Save current settings as preset"
        >
          Save
        </button>
      </div>
    </section>
  );
}
