import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import plotly.graph_objects as go
from PIL import Image

# Константы
SPEED_OF_LIGHT = 299792.458  # Скорость света в км/с

# Функция для загрузки данных TLE из файла
def load_tle_from_file(filename):
    tle_data = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines if line.strip()]
        if len(lines) % 2 != 0:
            raise ValueError(f"The TLE data file should contain an even number of lines. Found {len(lines)} lines.")
        for i in range(0, len(lines), 2):
            tle_data.append((lines[i], lines[i+1]))
    return tle_data

# Функция для преобразования TLE в объект спутника
def load_satellite_from_tle(tle_line1, tle_line2):
    satellite = Satrec.twoline2rv(tle_line1, tle_line2)
    return satellite

# Функция для расчета позиции спутника
def satellite_position(satellite, ts):
    jd, fr = jday(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)
    e, r, v = satellite.sgp4(jd, fr)
    if e != 0:
        raise RuntimeError(f"Error in SGP4 propagation: {e}")
    return r, v

# Функция для проверки видимости спутника
def is_satellite_visible(position_sat, observer_lat, observer_lon, observer_alt=0):
    R = 6371.0 + observer_alt  # Радиус Земли + высота наблюдателя
    observer_pos = R * np.array([
        np.cos(np.radians(observer_lat)) * np.cos(np.radians(observer_lon)),
        np.cos(np.radians(observer_lat)) * np.sin(np.radians(observer_lon)),
        np.sin(np.radians(observer_lat))
    ])
    
    relative_position = np.array(position_sat) - observer_pos
    distance = np.linalg.norm(relative_position)
    elevation = np.degrees(np.arcsin(relative_position[2] / distance))
    
    return elevation > 0, elevation, distance

# Функция для расчета прохождений
def calculate_passes(satellite, observer_lat, observer_lon, start_time, end_time, time_step=timedelta(minutes=1)):
    current_time = start_time
    passes = []

    while current_time <= end_time:
        position, _ = satellite_position(satellite, current_time)
        visible, elevation, distance = is_satellite_visible(position, observer_lat, observer_lon)

        if visible:
            entry_time = current_time
            max_elevation = elevation
            min_distance = distance

            while visible and current_time <= end_time:
                current_time += time_step
                position, _ = satellite_position(satellite, current_time)
                visible, elevation, distance = is_satellite_visible(position, observer_lat, observer_lon)
                if elevation > max_elevation:
                    max_elevation = elevation
                if distance < min_distance:
                    min_distance = distance

            exit_time = current_time
            passes.append((entry_time, exit_time, max_elevation, min_distance))
        else:
            current_time += time_step

    return passes

# Функция для расчета скорости наблюдателя
def observer_velocity(observer_lat, observer_lon):
    omega_earth = 7.2921159e-5  # Скорость вращения Земли в рад/с
    R = 6371.0  # Радиус Земли в км
    v = omega_earth * R * np.cos(np.radians(observer_lat))
    velocity = v * np.array([
        -np.sin(np.radians(observer_lon)),
        np.cos(np.radians(observer_lon)),
        0
    ])
    return velocity

# Функция для расчета относительной скорости
def relative_speed(satellite_velocity, observer_velocity):
    relative_velocity = np.array(satellite_velocity) - np.array(observer_velocity)
    speed = np.linalg.norm(relative_velocity)
    return speed

# Функция для расчета доплеровского сдвига
def doppler_shift(relative_velocity, frequency):
    doppler_shift = (frequency * relative_velocity) / SPEED_OF_LIGHT
    return doppler_shift

# Функция для расчета принимаемой мощности
def received_power(transmitted_power, gain_transmitter, gain_receiver, wavelength, distance):
    loss = (transmitted_power * gain_transmitter * gain_receiver * (wavelength ** 2)) / ((4 * np.pi * distance) ** 2)
    return loss

# Функция для расчета площади покрытия спутника
def calculate_coverage_area(position_sat, altitude):
    earth_radius = 6371.0  # Радиус Земли в км
    coverage_radius = np.sqrt((altitude + earth_radius)**2 - earth_radius**2)
    return coverage_radius

# Функция для преобразования широты и долготы в 3D координаты
def lat_lon_to_xyz(lat, lon, altitude=0):
    corrected_lon = lon - 180
    R = 6371.0 + altitude  # Радиус Земли + высота
    x = R * np.cos(np.radians(lat)) * np.cos(np.radians(corrected_lon))
    y = R * np.cos(np.radians(lat)) * np.sin(np.radians(corrected_lon))
    z = R * np.sin(np.radians(lat))
    return x, y, z

# Загрузка данных TLE
tle_data = load_tle_from_file('tle_data.txt')

# Создание объектов спутников
satellites = [load_satellite_from_tle(tle[0], tle[1]) for tle in tle_data]

# Получение координат наблюдателя
observer_lat = float(input("Введите широту наблюдателя: "))
observer_lon = float(input("Введите долготу наблюдателя: "))
# Для корректной работы тестов блок с кодом для ручного ввода надо закомменировать и использовать константы ниже
# observer_lat = 60
# observer_lon = 31

# Установка периода наблюдения
start_time = datetime.utcnow()
end_time = start_time + timedelta(hours=24)

# Свойства сигнала
nominal_frequency = float(input("Введите номинальную частоту (в МГц): "))
transmitted_power = float(input("Введите мощность передатчика (в Вт): "))
gain_transmitter = float(input("Введите коэффициент усиления передатчика: "))
gain_receiver = float(input("Введите коэффициент усиления приемника: "))
# Для корректной работы тестов блок с кодом для ручного ввода надо закомменировать и использовать константы ниже
# nominal_frequency = 1000000
# transmitted_power = 1000000
# gain_transmitter = 1000000
# gain_receiver = 1000000

# Расчет длины волны
wavelength = SPEED_OF_LIGHT / (nominal_frequency * 1e3)  # Перевод МГц в кГц для расчета

# Расчет прохождений для каждого спутника
all_passes = []
for satellite in satellites:
    passes = calculate_passes(satellite, observer_lat, observer_lon, start_time, end_time)
    all_passes.append(passes)
    for entry, exit, max_elevation, min_distance in passes:
        print(f"Спутник: {satellite.satnum}, Вход: {entry}, Выход: {exit}, Макс. угол возвышения: {max_elevation:.2f} градусов, Мин. расстояние: {min_distance:.2f} км")

# Настройка фигуры plotly
fig = go.Figure()

# Загрузка изображения поверхности
texture = np.asarray(Image.open('earth_wb.jpg')).T

# Массив для восстановления цвета из чб изображения
colorscale = [[0.0, 'rgb(30, 59, 117)'],
              [0.1, 'rgb(46, 68, 21)'],
              [0.2, 'rgb(74, 96, 28)'],
              [0.3, 'rgb(115,141,90)'],
              [0.4, 'rgb(122, 126, 75)'],
              [0.6, 'rgb(122, 126, 75)'],
              [0.7, 'rgb(141,115,96)'],
              [0.8, 'rgb(223, 197, 170)'],
              [0.9, 'rgb(237,214,183)'],
              [1.0, 'rgb(255, 255, 255)']]

# Создание координат сферы
N_lat = int(texture.shape[0])
N_lon = int(texture.shape[1])
u = np.linspace(0, 2 * np.pi, N_lat)
v = np.linspace(0, np.pi, N_lon)
x = 6371 * np.outer(np.cos(u), np.sin(v))
y = 6371 * np.outer(np.sin(u), np.sin(v))
z = 6371 * np.outer(np.ones(N_lat), np.cos(v))

# Добавление поверхности Земли
fig.add_trace(go.Surface(
    x=x, y=y, z=z,
    surfacecolor=texture,
    colorscale=colorscale,
    showscale=False,
    opacity=1.0
))

# Позиция наблюдателя
R = 6371.0
observer_pos = R * np.array([
    np.cos(np.radians(observer_lat)) * np.cos(np.radians(observer_lon)),
    np.cos(np.radians(observer_lat)) * np.sin(np.radians(observer_lon)),
    np.sin(np.radians(observer_lat))
])

# Скорость наблюдателя
observer_vel = observer_velocity(observer_lat, observer_lon)

# Построение орбит спутников и текущих позиций
colors = ['red', 'green', 'blue', 'purple', 'orange']
current_time = datetime.utcnow()
for i, satellite in enumerate(satellites):
    # Расчет орбиты
    positions = []
    velocities = []
    time_step = timedelta(minutes=3)
    time_range = [start_time + j * time_step for j in range(int((end_time - start_time) / time_step) + 1)]
    for ts in time_range:
        position, velocity = satellite_position(satellite, ts)
        positions.append(position)
        velocities.append(velocity)
    positions = np.array(positions)
    velocities = np.array(velocities)

    # Построение орбиты
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        mode='lines',
        line=dict(color=colors[i % len(colors)], width=2),
        name=f'Орбита спутника {satellite.satnum}'
    ))

    # Построение текущей позиции
    current_position, current_velocity = satellite_position(satellite, current_time)
    fig.add_trace(go.Scatter3d(
        x=[current_position[0]], y=[current_position[1]], z=[current_position[2]],
        mode='markers',
        marker=dict(size=6, color=colors[i % len(colors)], symbol='circle'),
        name=f'Текущая позиция спутника {satellite.satnum}'
    ))

    # Относительная скорость
    rel_speed = relative_speed(current_velocity, observer_vel)

    # Доплеровский сдвиг
    doppler = doppler_shift(rel_speed, nominal_frequency)

    # Принимаемая мощность
    distance_to_observer = np.linalg.norm(current_position - observer_pos)
    power = received_power(transmitted_power, gain_transmitter, gain_receiver, wavelength, distance_to_observer)

    # Вывод принимаемой мощности
    print(f"Спутник {satellite.satnum} - Принимаемая мощность у наблюдателя: {power:.3f} Вт")

    # Расчет и построение зоны покрытия прямо под спутником
    altitude = np.linalg.norm(current_position) - 6371.0
    coverage_radius = calculate_coverage_area(current_position, altitude)

    # Расчет точки подспутника
    sub_sat_lat = np.degrees(np.arcsin(current_position[2] / np.linalg.norm(current_position)))
    sub_sat_lon = np.degrees(np.arctan2(current_position[1], current_position[0]))

    coverage_lat_lon = []
    for angle in np.linspace(0, 360, 100):
        lat = np.degrees(np.arcsin(np.sin(np.radians(sub_sat_lat)) * np.cos(coverage_radius / 6371.0) +
                                   np.cos(np.radians(sub_sat_lat)) * np.sin(coverage_radius / 6371.0) * np.cos(
            np.radians(angle))))
        lon = -180 + sub_sat_lon + np.degrees(
            np.arctan2(np.sin(np.radians(angle)) * np.sin(coverage_radius / 6371.0) * np.cos(np.radians(sub_sat_lat)),
                       np.cos(coverage_radius / 6371.0) - np.sin(np.radians(sub_sat_lat)) * np.sin(np.radians(lat))))
        x, y, z = lat_lon_to_xyz(lat, lon)
        coverage_lat_lon.append((x, y, z))

    coverage_lat_lon = np.array(coverage_lat_lon)

    fig.add_trace(go.Scatter3d(
        x=coverage_lat_lon[:, 0], y=coverage_lat_lon[:, 1], z=coverage_lat_lon[:, 2],
        mode='lines',
        line=dict(color=colors[i % len(colors)], width=10, dash='dash'),
        name=f'Зона покрытия спутника {satellite.satnum}'
    ))

    fig.add_trace(go.Scatter3d(
        x=coverage_lat_lon[:, 0], y=coverage_lat_lon[:, 1], z=coverage_lat_lon[:, 2],
        mode='lines',
        line=dict(color=colors[i % len(colors)], width=10, dash='dash'),
        name=f'Относительная скорость: {rel_speed:.3f} км/с'
    ))

    fig.add_trace(go.Scatter3d(
        x=coverage_lat_lon[:, 0], y=coverage_lat_lon[:, 1], z=coverage_lat_lon[:, 2],
        mode='lines',
        line=dict(color=colors[i % len(colors)], width=10, dash='dash'),
        name=f'Доплер: {doppler:.5f} кГц'
    ))

# Настройка макета
fig.update_layout(
    title='Орбиты спутников и текущие позиции с доплеровским сдвигом и зонами покрытия',
    scene=dict(
        xaxis_visible=False,
        yaxis_visible=False,
        zaxis_visible=False,
        bgcolor='black'
    ),
    plot_bgcolor='black',
    paper_bgcolor='black',
    title_font_color='white',
    legend_font_color='white'
)

# Преобразование широты и долготы наблюдателя в 3D координаты
observer_xyz = lat_lon_to_xyz(observer_lat, observer_lon)

# Добавление маркера наблюдателя на макет
fig.add_trace(go.Scatter3d(
    x=[observer_xyz[0]],
    y=[observer_xyz[1]],
    z=[observer_xyz[2]],
    mode='markers',
    marker=dict(
        size=8,
        color='yellow',
        symbol='circle'
    ),
    name='Наблюдатель'
))

# Отображение фигуры
fig.show()

