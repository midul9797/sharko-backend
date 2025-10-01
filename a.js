const fs = require("fs");

// Read the GeoJSON file
const geojsonData = JSON.parse(
  fs.readFileSync("occurrence_data.geojson", "utf8")
);

// Remove properties field from each feature
if (geojsonData.features) {
  geojsonData.features.forEach((feature) => {
    if (feature.properties) {
      feature.properties = {};
    }
  });
}

// Optionally write the modified data back to a file
fs.writeFileSync("shark_track.geojson", JSON.stringify(geojsonData, null, 2));
