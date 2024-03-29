﻿//! \file color_correction.hpp
//! \brief Объявление функций, реализующих алгоритмы цветокоррекции.

#pragma once
#ifndef COLOR_CORRECTION_HPP
#define COLOR_CORRECTION_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


namespace cc {
    
    //! \brief Автоматическая настройка контраста и яркости
    //! \param[in] src - исходное изображение GRAY или BGR
    //! \param[out] dst - результирующее изображение
    //! \param[in] cut - процент пикселей, которые будут срезаны с левого и правого краев гистограммы (по умолчанию - 0)
    //! 
    //! Эта функция вычисляет гистограмму входного изображения, удаляет cut процентов самых светлых и самых темных пикселей из гистограммы
    //! и изменяет изображение, растягивая гистограмму таким образом, чтобы самый темный пиксель стал черным (0), а самый светлый стал белым (255).
    void autoContrast(const cv::Mat& src, cv::Mat& dst, float cut);

    //! \brief Изменение баланса белого в соответствии с алгоритмом Grey World
    //! \param[in] src - исходное изображение BGR
    //! \param[out] dst - результирующее изображение 
    //! 
    //! Эта функция вычисляет среднюю интенсивность по каждому каналу - avgR, avgG и avgB, среднюю интенсивность по всем каналам - avg,
    //! а затем масштабирует интенсивность пикселей в каждом канале по коэффициентам avg/avgR, avg/avgG и avg/avgB соответственно.
    void greyWorld(const cv::Mat& src, cv::Mat& dst);

}

#endif