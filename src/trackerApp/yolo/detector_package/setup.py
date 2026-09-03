from setuptools import find_packages, setup

package_name = 'detector_package'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tuo_nome',
    maintainer_email='you@example.com',
    description='Nodo di inferenza YOLOE (container yolo): DetectorNode + YoloEInference.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector_node = detector_package.detector_node:main',
        ],
    },
)
