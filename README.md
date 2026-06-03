# Yolo - Appose Fiji Plugin

This is a plugin to install and run [YOLO](https://docs.ultralytics.com/) for object detection from images in Fiji.

The plugin currently supports 2D images in grayscale or RGB.

This plugin used [Appose](https://github.com/apposed/appose), a framework that automatically installs python environement and allows python script execution with shared objects with Fiji.

## Plugin Installation

To install the plugin, download and copy the `.jar` file in the `plugins` directory of Fiji, and restart Fiji. The plugin should now be accessible in the plugin menu.

> [!NOTE]
> The python environment will be automatically installed in your home `.local\shared\appose` directory and activated from the plugin when needed.

## Usage

From Fiji, open the image that you want to process.
Launch the plugin from `Plugins>Yolo-Appose>Yolo Appose`.
An interface will pop-up to let you choose the parameters to run YOLO.
