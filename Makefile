# Définir le compilateur et les options
CXX = g++
CXXFLAGS = -std=c++17 -Iinclude

# Trouver tous les fichiers sources et générer les objets correspondants
SRCS = $(wildcard src/**/*.cpp)
OBJS = $(patsubst src/%.cpp, build/%.o, $(SRCS))

# Nom de l'exécutable cible
TARGET = project_metrics

# Règle principale
all: $(TARGET)

# Lier les objets pour créer l'exécutable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Règle spécifique pour ConfusionMatrix
build/evaluation/ConfusionMatrix.o: src/evaluation/ConfusionMatrix.cpp include/evaluation/ConfusionMatrix.h
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Règles pour compiler les autres fichiers sources
build/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Nettoyer les fichiers objets et l'exécutable
clean:
	rm -f $(OBJS) $(TARGET)
