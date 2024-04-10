# Лабораторные работы по курсу MLOps #1
## Lab 1
<details>

* Необходимо из создать простейший конвейер для автоматизации работы с моделью машинного обучения. 
* Отдельные этапы конвейера машинного обучения описываются в разных python–скриптах, которые потом соединяются в единую цепочку действий с помощью bash-скрипта.
* Все файлы необходимо разместить в подкаталоге lab1 корневого каталога

Этапы:
1. Создайте python-скрипт ([data_creation.py](https://github.com/ANGorbachev/MLOps_1/blob/main/lab1/data_creation.py)), который создает различные наборы данных, описывающие некий процесс (например, изменение дневной температуры). Таких наборов должно быть несколько, в некоторые данные можно включить аномалии или шумы. 
Часть наборов данных должны быть сохранены в папке “train”, другая часть в папке “test”. Одним из вариантов выполнения этого этапа может быть скачивание набора данных из сети, и разделение выборки на тестовую и обучающую. Учтите, что файл должен быть доступен и методы скачивания либо есть в ubuntu либо устанавливаются через pip в файле pipeline.sh
2. Создайте python-скрипт ([data_preprocessing.py](https://github.com/ANGorbachev/MLOps_1/blob/main/lab1/model_preprocessing.py)), который выполняет предобработку данных, например, с помощью sklearn.preprocessing.StandardScaler. Трансформации выполняются и над тестовой и над обучающей выборкой. 
3. Создайте python-скрипт ([model_preparation.py](https://github.com/ANGorbachev/MLOps_1/blob/main/lab1/model_preparation.py)), который создает и обучает модель машинного обучения на построенных данных из папки “train”. Для сохранения модели в файл можно воспользоваться [pickle](https://docs.python.org/3/library/pickle.html) (см. [пример](https://rukovodstvo.net/posts/id_1322/))
4. Создайте python-скрипт ([model_testing.py](https://github.com/ANGorbachev/MLOps_1/blob/main/lab1/model_testing.py)), проверяющий модель машинного обучения на построенных данных из папки “test”.
5. Напишите bash-скрипт ([pipeline.sh](https://github.com/ANGorbachev/MLOps_1/blob/main/lab1/pipeline.sh)), последовательно запускающий все python-скрипты. При необходимости усложните скрипт. В результате выполнения скрипта на терминал в стандартный поток вывода печатается одна строка с оценкой метрики на вашей модели, например:

Для запуска выполнить bash скрипт
```shell
chmod +x pipeline.sh
./pipeline.sh
```

</details>

## Lab 2
<details>

### Цель задания
В практическом задании по этому модулю вам нужно разработать собственный конвейер 
автоматизации для проекта машинного обучения. Конвейер должен быть аналогичен тому, 
который мы рассмотрели в последнем юните этого модуля.


### Содержание задания
Для этого вам понадобится виртуальная машина с установленным Jenkins, python и 
необходимыми библиотеками. В ходе выполнения практического задания вам необходимо 
автоматизировать сбор данных, подготовку датасета, обучение модели и работу модели.


### Этапы
1. Развернуть сервер с Jenkins, установить необходимое программное обеспечение для работы 
над созданием модели машинного обучения.
2. Выбрать способ получения данных ([create_dataset.py](https://github.com/ANGorbachev/MLOps_1/blob/main/lab2/create_dataset.py)).
3. Провести обработку данных, выделить важные признаки, сформировать датасеты для тренировки и тестирования модели, сохранить ([data_preprocessing.py](https://github.com/ANGorbachev/MLOps_1/blob/main/lab2/data_preprocessing.py))).
4. Создать и обучить на тренировочном датасете модель машинного обучения, сохранить в pickle или аналогичном формате ([model_training.py](https://github.com/ANGorbachev/MLOps_1/blob/main/lab2/model_training.py)).
5. Загрузить сохраненную модель на предыдущем этапе и проанализировать ее качество на тестовых данных ([model_testing.py](https://github.com/ANGorbachev/MLOps_1/blob/main/lab2/model_testing.py)).
6. Реализовать задания и конвейер. Связать конвейер с системой контроля версий. Сохранить конвейер ([Jenkinsfile](https://github.com/ANGorbachev/MLOps_1/blob/main/lab2/Jenkinsfile)).

</details>
