#Tasks
# Get and set pixels of image
Что из себя представляет изображение.
Это матрица по ширине и высоте изображения в пикселях и глубиной в 3.
![alt text](rgb.png)
Можно представить схемотично в виде
![alt text](coords.png)
[Пример работы](https://pyimagesearch.com/2021/01/20/opencv-getting-and-setting-pixels/)

Возьмем любое изображение из папки и попробуем получить его цифровое представление.
    
    image[100,100]
После чего попробуем его изменить по пиксельно.
    
    image[100,100] = [255, 255, 255]
# Copying images
Понятно, что можно работать с базовыми понятиями OS, и использовать их. Но можно и работать с openCV
    
    imwrite("test.png", image)

$/int{0}$

# Grayscale image
Длины волн, которые может видеть глаз человека
![чувствительность глаза](sensitivity.png)
Обычная линейка цветов при переводе в черно-белое изображние 
![цветная полоса](colorbar.png)
    
    K=(R+G+B)/3
![чб полоса](avggray.jpg)
Но интуитивный формат не очень читаем и понятен 

    Y' = 0.299 R' + 0.587 G' + .114 B'
![gray](gray.png)
# Shifting the image colours

# Clamping the image values

# RGB to Hue, Saturation, Value
![convert](convert.png)
![](eq.svg)
# HSV to RGB

# Change Saturation