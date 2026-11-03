// An "upload mesh" button under the model_file picker of LATO.2 Low-Poly Remesh: the file goes to input/3d (where
// Load 3D puts its uploads) and is selected. ComfyUI draws such a button only for its own Load 3D node.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODES = new Set(["LATO2Remesh"]);
const ACCEPT = ".glb,.gltf,.obj,.ply,.stl";

async function upload(file) {
  const body = new FormData();
  body.append("image", file);
  body.append("type", "input");
  body.append("subfolder", "3d");
  body.append("overwrite", "true");
  const response = await api.fetchApi("/upload/image", { method: "POST", body });
  if (response.status !== 200) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  const data = await response.json();
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

app.registerExtension({
  name: "LATO2.Upload",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.has(nodeData.name)) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      const picker = this.widgets?.find((w) => w.name === "model_file");
      if (!picker || this.widgets.some((w) => w.type === "button" && /upload/i.test(w.name))) return result;
      const chooser = document.createElement("input");
      chooser.type = "file";
      chooser.accept = ACCEPT;
      chooser.style.display = "none";
      document.body.append(chooser);
      chooser.onchange = async () => {
        const file = chooser.files?.[0];
        chooser.value = "";
        if (!file) return;
        try {
          const name = await upload(file);
          if (!picker.options.values.includes(name)) picker.options.values.push(name);
          picker.value = name;
          picker.callback?.(name);
          this.setDirtyCanvas(true, true);
        } catch (error) {
          alert(`Mesh upload failed: ${error.message}`);
        }
      };
      const button = this.addWidget("button", "upload mesh", null, () => chooser.click());
      button.serialize = false;
      // Right under the picker.
      this.widgets.splice(this.widgets.indexOf(button), 1);
      this.widgets.splice(this.widgets.indexOf(picker) + 1, 0, button);
      const onRemoved = this.onRemoved;
      this.onRemoved = function () {
        chooser.remove();
        return onRemoved?.apply(this, arguments);
      };
      return result;
    };
  },
});
