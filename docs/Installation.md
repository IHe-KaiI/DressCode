# Installation Instructions

## Dependencies

* Autodesk Maya 2022+ with Arnold
  
* [Qualoth](http://www.fxgear.net/vfx-software?locale=en), version 4.7.7+ (with Maya 2022 support)

    >We use Maya 2022 and Qualoth 4.7 in development.
### Python Environment in Maya

To install libs in Maya using pip, ``` YOUR_PATH_TO_MAYAPY -m pip install PACKEGE_NAME```, e.g., ``` "C:\Program Files\Autodesk\Maya2022\bin\mayapy.exe" -m pip install numpy```.

* numpy
* scipy
* [svglib](https://pypi.org/project/svglib/)
* [svgwrite](https://pypi.org/project/svgwrite/)
* psutil
* pyyaml
* wmi
    * May require pypiwin32
    * [Troubleshooting 'DLL load failed while importing win32api'](https://stackoverflow.com/questions/58612306/how-to-fix-importerror-dll-load-failed-while-importing-win32api) error on Win

<details>
    <summary> <b>NOTE: Lib versions used in development</b></summary>
    python==3.7.7
    numpy==1.21.6
    scipy==1.7.3
    svglib==1.5.1
    svgwrite==1.4.3
    psutil==5.9.6
    wmi=1.5.1
</details>



## Acknowledgment

- [Dataset of 3D Garments with Sewing patterns](https://github.com/maria-korosteleva/Garment-Pattern-Generator/tree/master)
- [Sewformer](https://github.com/sail-sg/sewformer)