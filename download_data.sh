# NASA's Battery Dataset 
# B. Saha and K. Goebel (2007). "Battery Data Set", 
# NASA Ames Prognostics Data Repository (http://ti.arc.nasa.gov/project/prognostic-data-repository), 
# NASA Ames Research Center, Moffett Field, CA"

unzd() {
  target="${1%.zip}"
  unzip "$1" -d "${target##*/}"
}

cd ./data

echo "Downloading Battery Data Set 1..."
wget https://ti.arc.nasa.gov/m/project/prognostic-repository/BatteryAgingARC-FY08Q4.zip
unzd BatteryAgingARC-FY08Q4.zip
rm BatteryAgingARC-FY08Q4.zip

echo "Downloading Battery Data Set 2..."
wget https://ti.arc.nasa.gov/m/project/prognostic-repository/BatteryAgingARC_25_26_27_28_P1.zip 
unzd BatteryAgingARC_25_26_27_28_P1.zip 
rm BatteryAgingARC_25_26_27_28_P1.zip 

echo "Downloading Battery Data Set 3..."
wget https://ti.arc.nasa.gov/m/project/prognostic-repository/BatteryAgingARC_25-44.zip
unzd BatteryAgingARC_25-44.zip
rm BatteryAgingARC_25-44.zip

echo "Downloading Battery Data Set 4..."
wget https://ti.arc.nasa.gov/m/project/prognostic-repository/BatteryAgingARC_45_46_47_48.zip
unzd BatteryAgingARC_45_46_47_48.zip
rm BatteryAgingARC_45_46_47_48.zip

echo "Downloading Battery Data Set 5..."
wget https://ti.arc.nasa.gov/m/project/prognostic-repository/BatteryAgingARC_49_50_51_52.zip
unzd BatteryAgingARC_49_50_51_52.zip
rm BatteryAgingARC_49_50_51_52.zip

echo "Downloading Battery Data Set 6..."
wget https://ti.arc.nasa.gov/m/project/prognostic-repository/BatteryAgingARC_53_54_55_56.zip
unzd BatteryAgingARC_53_54_55_56.zip
rm BatteryAgingARC_53_54_55_56.zip

cd -
