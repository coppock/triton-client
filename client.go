package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"
)

var authority = flag.String("a", "localhost:8000",
	"Triton inference server authority [localhost:8000]")
var rate float64
var model string

type Config struct {
	MaxBatchSize int `json:"max_batch_size"`
	Input        []struct {
		Name     string
		Datatype string `json:"data_type"`
		Dims     []int
		Optional bool
	}
}

type Infer struct {
	Inputs []Input `json:"inputs"`
}

type Input struct {
	Name     string          `json:"name"`
	Datatype string          `json:"datatype"`
	Shape    []int           `json:"shape"`
	Data     json.RawMessage `json:"data"`
}

func main() {
	flag.Parse()
	model = flag.Args()[0]
	r, err := strconv.ParseFloat(flag.Args()[1], 64)
	if err != nil {
		panic(err)
	}
	rate = r

	var config Config
	config.Init()

	for t := range load(config) {
		fmt.Println(float64(t.UnixMicro())/1e6, time.Since(t).Seconds())
	}
}

func (c *Config) Init() {
	resp, err := http.Get(fmt.Sprintf("http://%s/v2/models/%s/config",
		*authority, model))
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		panic(err)
	}
	if resp.StatusCode != 200 {
		panic(fmt.Sprintf("%s\n%s\n", resp.Status, body))
	}
	err = json.Unmarshal(body, c)
	if err != nil {
		panic(err)
	}
}

func load(c Config) <-chan time.Time {
	after := make(chan time.Time, 10)
	go func() {
		d := time.Duration(float64(time.Second) / rate)
		for t := range time.Tick(d) {
			go infer(c, t, after)
		}
	}()
	return after
}

func infer(c Config, t time.Time, after chan time.Time) {
	var inputs []Input
	for _, input := range c.Input {
		if !input.Optional {
			shape := input.Dims
			if c.MaxBatchSize >= 1 {
				shape = []int{1}
				shape = append(shape, input.Dims...)
			}
			inputs = append(inputs, NewInput(
				input.Name,
				input.Datatype,
				shape,
			))
		}
	}
	body, err := json.Marshal(Infer{inputs})
	if err != nil {
		panic(err)
	}
	resp, err := http.Post(fmt.Sprintf("http://%s/v2/models/%s/infer",
		*authority, model), "application/json", bytes.NewReader(body))
	if err != nil {
		log.Println("Post")
		return
	}
	defer resp.Body.Close()
	body, err = io.ReadAll(resp.Body)
	if err != nil {
		panic(err)
	}
	if resp.StatusCode != 200 {
		panic(fmt.Sprintf("%s\n%s\n", resp.Status, body))
	}
	after <- t
}

func NewInput(name string, datatype string, dims []int) Input {
	datatype = strings.TrimPrefix(datatype, "TYPE_")

	length := 1
	for _, dim := range dims {
		length *= dim
	}

	var i Input
	i.Name = name
	i.Datatype = datatype
	i.Shape = dims

	switch datatype {
	case "FP32":
		b, err := json.Marshal(make([]float32, length))
		if err != nil {
			panic(err)
		}
		i.Data = b
	default:
		panic(fmt.Sprintf("NewInput: datatype %s not supported",
			datatype))
	}
	return i
}
