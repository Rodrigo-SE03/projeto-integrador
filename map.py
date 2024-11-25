import json
import folium
import streamlit as st
from streamlit_folium import st_folium
from shapely.geometry import Point, shape



def render_map(zoom_level=6):
    """
    Renderiza um mapa com funcionalidade de marcador único e destaca a borda do estado de Goiás.
    """

    configs = {
        "cor_borda": "orange",  # Cor da borda de Goiás
        "espessura_borda": 2,   # Espessura da borda de Goiás
    }

    # Coordenadas iniciais para centralizar o mapa
    goias_center = [-16.6869, -49.2648]  # Coordenadas aproximadas de Goiânia, GO

    # Inicializar coordenadas do marcador (se houver)
    if "marker_coords" not in st.session_state:
        st.session_state["marker_coords"] = None

    # Criar o mapa
    mapa = folium.Map(
        location=goias_center,
        zoom_start=zoom_level,
        dragging=False,  # Desativa arrastar o mapa
        zoom_control=False,  # Remove os controles de zoom
        scrollWheelZoom=False,  # Desativa zoom pela roda do mouse
        touchZoom=False,  # Desativa zoom por toque
    )

    # Carregar o GeoJSON de Goiás
    with open("goias_geo.json", "r", encoding="utf-8") as f:
        goias_geojson = json.load(f)

    # Criar o polígono do estado de Goiás
    goias_polygon = shape(goias_geojson["features"][0]["geometry"])

    # Adicionar a borda de Goiás
    folium.GeoJson(
        goias_geojson,
        name="Limite de Goiás",
        style_function=lambda x: {
            "color": configs["cor_borda"],  # Cor da borda
            "weight": configs["espessura_borda"],      # Espessura da borda
            "fillOpacity": 0  # Transparência da área preenchida
        }
    ).add_to(mapa)

    # Adicionar marcador único, se existir
    if st.session_state["marker_coords"] is not None:
        folium.Marker(
            location=st.session_state["marker_coords"],
            popup=f"Lat: {st.session_state['marker_coords'][0]}, Lng: {st.session_state['marker_coords'][1]}",
        ).add_to(mapa)

    # Renderizar o mapa no Streamlit
    st_data = st_folium(mapa, width=700, height=500)

    # Capturar as coordenadas do clique
    if st_data["last_clicked"] is not None:
        coords = [st_data["last_clicked"]["lat"], st_data["last_clicked"]["lng"]]
        st.session_state["marker_coords"] = coords  # Atualiza as coordenadas do marcador

        # Atualiza o mapa com o novo marcador
        mapa = folium.Map(
            location=goias_center,
            zoom_start=zoom_level,
            dragging=False,
            zoom_control=False,
            scrollWheelZoom=False,
            touchZoom=False,
        )

        # Recarregar a borda de Goiás
        folium.GeoJson(
            goias_geojson,
            name="Limite de Goiás",
            style_function=lambda x: {
                "color": configs["cor_borda"],  # Cor da borda
                "weight": configs["espessura_borda"],      # Espessura da borda
                "fillOpacity": 0
            }
        ).add_to(mapa)

        # Adicionar o novo marcador
        folium.Marker(
            location=coords,
            popup=f"Lat: {coords[0]}, Lng: {coords[1]}",
        ).add_to(mapa)

        # Renderizar novamente o mapa
        st_folium(mapa, width=700, height=500)

        # Verificar se o ponto está dentro de Goiás
        point = Point(coords[1], coords[0])  # (longitude, latitude)
        if goias_polygon.contains(point) == False:
            st.session_state["marker_coords"] = [0,0]

    # Retorna as coordenadas clicadas
    return st.session_state["marker_coords"]
