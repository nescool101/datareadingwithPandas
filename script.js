console.log("Inicia ejecución del script"); // Mensaje por consola de inicio de ejecución del script
d3.json("tiendas.json").then(function (dataset) { //Cargue del archivo alumnos_master en formato json

    // Dimensiones del gráfico en pixeles.
    var width = 800; // Ancho  
    var height = 600; // Alto

    // Se establecen los márgenes del gráfico en pixeles
    var margenes = {  
        arriba: 60,
        abajo:35,
        izquierda:40,
        derecha:50
    }

    // Selecciona el elemento <body> del documento y agrega un encabezado <h1> con un texto específico.
    d3.select("body")
        .append("h1") 
        .text("Ranking de Ventas de Tiendas Online")
        .style("color", "black");

    // CONFIGURACIÓN DE LOS EJES VERTICAL Y HORIZONTAL
    // Se crea la escala del eje vertical (eje y)
    var escala_ejeY = d3.scaleLinear()
        .domain([0,10]) // Se usará un dominio fijo
        .range([height - margenes.abajo, margenes.arriba]) //se establece la ubicación del eje vertical

    // Escala para eje horizontal (eje x)
    var escala_ejeX = d3.scaleLinear()
        .domain([0, 1000]) //Dominio fijo para eje x
        .range([margenes.izquierda, width-margenes.derecha]) // Ubicación del eje x


    // CONFIGURACIÓN DEL TAMAÑO Y COLOR DE LOS CÍRCULOS
    // Tamaño de los Circulos del gráfico de dispersión
    var TamanoCirculo = d3.scaleLinear()
        .domain(d3.extent(dataset, function(datos){return datos.Venta})) //La variable venta definirá el tamaño de los círculos
        .range([10, 50]) //Rango dentro del cual se establece el tamaño del círculo

    // Definir color de los círculos
    var ColorCirculo = d3.scaleLinear()
        .domain(d3.extent(dataset, function(datos){return datos.Ranking})) // La variable ranking define el color del círculo
        .range(["green", "red"]) // Colores de los círculos. Rojo para Ventas bajas y verde para Ventas altas
    
    
    // CONFIGURACIÓN DEL ELEMENTO SVG QUE CONTENDRÁ LA GRÁFICA REALIZADA CON D3
    // Creación del lienzo SVG para generar allí el gráfico con D3
    var lienzoSVG = d3.select("body")
        .append("svg")
        .attr("id", "miSVG_D3")
        .attr("width", width)
        .attr("height", height)

    // Creación de los círculos del gráfico dentro del lienzo SVG
    lienzoSVG.selectAll("circle") 
        .data(dataset)  // Carga los datos del json en el lienzo SVG
        .join("circle") // Crea un cículo para cada elemento del json
        .attr("r", function(datos){return TamanoCirculo(datos.Venta)}) // Define el radio del círculo con base en la variable TamanoCirculo
        .attr("cx", function(datos){return escala_ejeX(datos.Venta)})     // Ubicación horizontal del círculo con base en la variable escala_ejeX
        .attr("cy", function(datos){return escala_ejeY(datos.Ranking)})  // Ubicación vertical del círculo con base en la variable escala_ejeY
        .attr("fill", function(datos){return ColorCirculo(datos.Ranking)})  // Define el color del círculo con base en la variable ColorCirculo
        .on("mouseover", (event, datos) => generarTooltip(event, datos))  // Muestra el tooltip al pasar el mouse sobre el círculo
        .on("mouseout", borrarTooltip) // desaparece el tooltip cuando el mouse sale del círculo

    // Creación de un elemento <div> para el tooltip para mostrar información adicional (nombre, nota y ranking)
    var tooltip = d3.select("body")
        .append("div")
        .attr("class", "tooltip")
            
    // configuración de animación del eje Y
    var ejeY = d3.axisLeft(escala_ejeY)
    lienzoSVG.append("g") // genera la animación sobre el eje
        .attr("transform", "translate(" + margenes.izquierda + ",0)")
        .call(ejeY)
        .attr("opacity", 0)
        .transition() // Aplicamos la transición
        .duration(0) // Duración en milisegundos
        .attr("opacity", 1);

    // configuración de animación del eje X
    var ejeX = d3.axisBottom(escala_ejeX)
    lienzoSVG.append("g")
        .attr("transform", "translate(0," + (height - margenes.abajo) + ")")
        .call(ejeX)
        .attr("opacity", 0)
        .transition() // Aplicamos la transición
        .duration(0) // Duración en milisegundos
        .attr("opacity", 1);

    function borrarTooltip() {
        tooltip.style("opacity", 0)
    }

    function generarTooltip(event, datos) {
        tooltip.text("Tienda: "+ datos.Tienda + " Venta: " + datos.Venta)
            .style("top", event.y + "px")
            .style("left", event.x + "px")
            .style("opacity", 1)
    }

})
