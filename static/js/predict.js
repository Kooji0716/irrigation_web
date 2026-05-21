function setValue(id, value) {
    document.getElementById(id).value = value;
}

function loadExample() {
    const example = document.getElementById("exampleSelect").value;

    if (example === "") {
        alert("請先選擇一組範例情境");
        return;
    }

    if (example === "dry") {
        setValue("Soil_Type", "Sandy");
        setValue("Soil_pH", 6.20);
        setValue("Soil_Moisture", 15.00);
        setValue("Organic_Carbon", 0.60);
        setValue("Electrical_Conductivity", 1.20);

        setValue("Temperature_C", 34.00);
        setValue("Humidity", 38.00);
        setValue("Rainfall_mm", 150.00);
        setValue("Sunlight_Hours", 10.50);
        setValue("Wind_Speed_kmh", 28.00);

        setValue("Crop_Type", "Rice");
        setValue("Crop_Growth_Stage", "Flowering");
        setValue("Season", "Kharif");
        setValue("Region", "South");

        setValue("Irrigation_Type", "Drip");
        setValue("Water_Source", "Groundwater");
        setValue("Field_Area_hectare", 2.50);
        setValue("Mulching_Used", "No");
        setValue("Previous_Irrigation_mm", 20.00);
    }

    if (example === "normal") {
        setValue("Soil_Type", "Loamy");
        setValue("Soil_pH", 6.80);
        setValue("Soil_Moisture", 35.00);
        setValue("Organic_Carbon", 1.10);
        setValue("Electrical_Conductivity", 1.80);

        setValue("Temperature_C", 26.00);
        setValue("Humidity", 62.00);
        setValue("Rainfall_mm", 850.00);
        setValue("Sunlight_Hours", 7.50);
        setValue("Wind_Speed_kmh", 14.00);

        setValue("Crop_Type", "Wheat");
        setValue("Crop_Growth_Stage", "Vegetative");
        setValue("Season", "Rabi");
        setValue("Region", "Central");

        setValue("Irrigation_Type", "Canal");
        setValue("Water_Source", "River");
        setValue("Field_Area_hectare", 1.80);
        setValue("Mulching_Used", "Yes");
        setValue("Previous_Irrigation_mm", 45.00);
    }

    if (example === "wet") {
        setValue("Soil_Type", "Clay");
        setValue("Soil_pH", 7.10);
        setValue("Soil_Moisture", 58.00);
        setValue("Organic_Carbon", 1.40);
        setValue("Electrical_Conductivity", 2.10);

        setValue("Temperature_C", 21.00);
        setValue("Humidity", 85.00);
        setValue("Rainfall_mm", 1900.00);
        setValue("Sunlight_Hours", 5.50);
        setValue("Wind_Speed_kmh", 6.00);

        setValue("Crop_Type", "Sugarcane");
        setValue("Crop_Growth_Stage", "Harvest");
        setValue("Season", "Kharif");
        setValue("Region", "East");

        setValue("Irrigation_Type", "Rainfed");
        setValue("Water_Source", "Rainwater");
        setValue("Field_Area_hectare", 3.00);
        setValue("Mulching_Used", "Yes");
        setValue("Previous_Irrigation_mm", 80.00);
    }
}

//後面更改的功能
//如果使用者選「非神經網路模型」→ 第二個下拉選單出現 5 個模型
//如果使用者選「神經網路模型」→ 第二個下拉選單只出現 Keras / TensorFlow MLP

function updateModelOptions() {
    const modelGroup = document.getElementById("model_group").value;
    const modelType = document.getElementById("model_type");

    modelType.innerHTML = "";

    if (modelGroup === "sklearn") {
        const options = [
            ["lr", "Logistic Regression"],
            ["rf", "Random Forest"],
            ["rf_new", "Random Forest New"],
            ["xgb", "XGBoost"],
            ["hgbc", "HistGradientBoosting"]
        ];

        options.forEach(function(item) {
            const option = document.createElement("option");
            option.value = item[0];
            option.textContent = item[1];
            modelType.appendChild(option);
        });
    }

    if (modelGroup === "tensorflow") {
        const option = document.createElement("option");
        option.value = "nn";
        option.textContent = "Keras / TensorFlow MLP";
        modelType.appendChild(option);
    }
}