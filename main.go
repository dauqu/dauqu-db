package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

type Embedding struct {
	Vector []float64
}

type EmbeddingDB struct {
	Embeddings map[string]Embedding
}

func NewEmbeddingDB() *EmbeddingDB {
	return &EmbeddingDB{
		Embeddings: make(map[string]Embedding),
	}
}

func (db *EmbeddingDB) Add(key string, vector []float64) {
	db.Embeddings[key] = Embedding{
		Vector: vector,
	}
}

func (db *EmbeddingDB) Get(key string) ([]float64, bool) {
	embedding, ok := db.Embeddings[key]
	if !ok {
		return nil, false
	}
	return embedding.Vector, true
}

func (db *EmbeddingDB) CosineSimilarity(key1, key2 string) (float64, error) {
	vec1, ok := db.Get(key1)
	if !ok {
		return 0, fmt.Errorf("embedding not found for key: %s", key1)
	}

	vec2, ok := db.Get(key2)
	if !ok {
		return 0, fmt.Errorf("embedding not found for key: %s", key2)
	}

	if len(vec1) != len(vec2) {
		return 0, fmt.Errorf("vectors have different dimensions")
	}

	dotProduct := 0.0
	magnitudeVec1 := 0.0
	magnitudeVec2 := 0.0
	for i := 0; i < len(vec1); i++ {
		dotProduct += vec1[i] * vec2[i]
		magnitudeVec1 += vec1[i] * vec1[i]
		magnitudeVec2 += vec2[i] * vec2[i]
	}

	magnitudeVec1 = math.Sqrt(magnitudeVec1)
	magnitudeVec2 = math.Sqrt(magnitudeVec2)

	if magnitudeVec1 == 0 || magnitudeVec2 == 0 {
		return 0, fmt.Errorf("zero vector")
	}

	cosineSimilarity := dotProduct / (magnitudeVec1 * magnitudeVec2)
	return cosineSimilarity, nil
}

func main() {
	// Create a new embedding database
	db := NewEmbeddingDB()

	// Open the text file for reading
	file, err := os.Open("embeddings.txt")
	if err != nil {
		fmt.Println("Error opening file:", err)
		return
	}
	defer file.Close()

	// Read embedding data from the text file
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, " ")
		if len(parts) >= 2 {
			key := parts[0]
			var vector []float64
			for _, valStr := range parts[1:] {
				val, err := strconv.ParseFloat(valStr, 64)
				if err != nil {
					fmt.Printf("Error parsing embedding value: %s\n", valStr)
					continue
				}
				vector = append(vector, val)
			}
			if len(vector) > 0 {
				db.Add(key, vector)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Println("Error reading file:", err)
		return
	}

	// Get embedding for a specific key
	embedding, ok := db.Get("banana")
	if ok {
		fmt.Println("Embedding for 'banana':", embedding)
	} else {
		fmt.Println("Embedding not found for 'banana'")
	}

	// Calculate cosine similarity between two embeddings
	cosineSimilarity, err := db.CosineSimilarity("apple", "orange")
	if err == nil {
		fmt.Println("Cosine similarity between 'apple' and 'orange':", cosineSimilarity)
	} else {
		fmt.Println("Error calculating cosine similarity:", err)
	}

	// Prompt user for input to get cosine similarity between two embeddings
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Println("Enter two words to calculate cosine similarity (e.g. apple orange):")
		fmt.Print("> ")
		input, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println("Error reading input:", err)
			continue
		}
		input = strings.TrimSpace(input)
		parts := strings.Split(input, " ")
		if len(parts) != 2 {
			fmt.Println("Please enter exactly two words separated by space")
			continue
		}
		word1 := parts[0]
		word2 := parts[1]
		cosineSimilarity, err := db.CosineSimilarity(word1, word2)
		if err == nil {
			fmt.Printf("Cosine similarity between '%s' and '%s': %f\n", word1, word2, cosineSimilarity)
		} else {
			fmt.Println("Error calculating cosine similarity:", err)
		}
	}
}
