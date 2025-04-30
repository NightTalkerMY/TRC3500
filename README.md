# TRC3500 Group Project

## Overview
This project consists of three main components: a **Soil Moisture Sensor**, a **Pin Drop Detector**, and a **Breath Rate Estimator**. Each system is designed to demonstrate various sensor applications, signal processing techniques, and embedded system integration.

## 1. Soil Moisture Sensor
### Description
The soil moisture sensor is designed to measure the water content in soil using a capacitive sensing method. The sensor consists of stainless electrodes to detect moisture levels and uses an **STM32 microcontroller** for signal processing.

### Features
- Capacitive sensing with stainless electrodes
- Signal conditioning using a **TLV9054 op-amp**
- Data acquisition via STM32 microcontroller
- Calibration for repeatability and accuracy

### Applications
- Smart agriculture
- Automated irrigation systems
- Soil health monitoring

## 2. Pin Drop Detector
### Description
The pin drop detector is an ultra-sensitive acoustic detection system capable of detecting minimal sound events, such as a pin drop, using a piezo sensor, advanced signal processing algorithms and machine learning (Random Forest Algorithm) 

### Features
- High-gain piezo sensor for detecting low-intensity sounds
- Signal processing using STM32 microcontroller
- Machine Learning and threshold-based event detection

### Applications
- Security and surveillance
- Sound-based event detection

## 3. Breath Rate Estimator
### Description
The breath rate estimator is a biomedical application that uses a non-contact method to estimate a person's breathing rate. This system uses a **MEMS microphone** to capture breath sounds and analyze periodic patterns.

### Features
- Non-contact breath monitoring
- Real-time data acquisition and processing
- Frequency analysis for breath rate estimation

### Applications
- Health monitoring
- Sleep apnea detection
- Respiratory diagnostics

## Hardware Components
- STM32 Microcontroller
- TLV9054 Op-Amp (for soil moisture sensor signal conditioning)
- Piezo Sensor (for pin drop detection and breath rate estimation)
- Stainless steel electrodes (for soil moisture sensing)
- MEMS microphone (for breath rate monitoring)

## Software & Algorithms
- Embedded C programming for STM32
- Signal conditioning and filtering techniques
- Data analysis and real-time monitoring

## Future Improvements
- Integration with a wireless data transmission system
- Machine learning for more accurate classification
- Improved noise reduction techniques for enhanced sensitivity

---

This project demonstrates multi-disciplinary applications of sensors in **agriculture**, **security**, and **biomedical engineering**. Each subsystem contributes to understanding real-world signal processing challenges and embedded system design.

